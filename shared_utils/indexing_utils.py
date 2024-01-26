import torch
import h5py
import faiss
import json
import os
import random
import logging
import sys
import subprocess
import torch.distributed as dist
from .distributed_utils import is_main, data_sharding, all_gather_items, dist_print
from pyserini.search import SimpleSearcher
import numpy as np


logger = logging.getLogger(__name__)


class DocumentCollection:
    
    def __init__(self,
                 data_path,
                 max_passage_length=384,
                 seed=42):
        self.data_path = data_path
        self.max_passage_length = max_passage_length
        self.file = None
        self.length = 0
        
        if os.path.exists(data_path):
            self.data_path = data_path
            self.file = h5py.File(data_path, "r")
            self.length = len(self.file["data"])

        self.rng = random.Random(seed)
        
    def write_h5(self,
                 passage_files,
                 ctx_tokenizer,
                 logging_step=100):

        file = h5py.File(f"{self.data_path}", "w")
        f = file.create_dataset("data",
                                dtype=h5py.string_dtype(),
                                shape=(100, ),
                                chunks=True,
                                maxshape=(None,),
                                compression="gzip")

        instances = []
        h5_index = 0
        for passage_file in passage_files:
            for idx, line in enumerate(open(passage_file, "r", encoding="utf-8")):
                example = json.loads(line)
                text = example["contents"]
                example_id = example["id"]
                encoded_example = ctx_tokenizer(text,
                                                padding="max_length",
                                                truncation=True,
                                                max_length=self.max_passage_length)
                i = {"input_id": encoded_example["input_ids"], "text": text, "id": example_id}
                instances.append(json.dumps(i))
                h5_index += 1

                if h5_index % 1000 == 0:
                    f.resize(h5_index, axis=0)
                    f[h5_index-1000:h5_index] = instances
                    instances = []

                if idx % logging_step == 0:
                    logger.info(f"Write passage data.h5 ... [{h5_index}]")

        if len(instances) > 0:
            f.resize(h5_index, axis=0)
            f[h5_index - len(instances):h5_index] = instances
            instances = []
        file.close()
        logger.info(f"Write passage data.h5 done!")

    def __len__(self):
        return self.length

    def get_data(self, id_):
        return json.loads(self.file["data"][id_])["input_id"]
    
    def get_text(self, id_):
        return json.loads(self.file["data"][id_])["text"]
    
    def get_pid(self, id_):
        return json.loads(self.file["data"][id_])["id"]

    def negative_sampling(self, negative_ids, k):
        negative_ids = self.rng.sample(negative_ids, k)
        return [self.get_data(i) for i in negative_ids]


class FaissIndex:
    
    def __init__(self, dim, path=None, device=-1, ann_search=False):
        self.dim = dim
        self.index = faiss.IndexFlatIP(self.dim)
        self.device = device
        self.ann_search = ann_search
        if self.ann_search:
            index = faiss.IndexHNSWFlat(self.dim, 512, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 128
            index.hnsw.efConstruction = 256
            self.index = index
            self.buffer = []

    def __len__(self):
        return self.index.ntotal
    
    def construct_hnsw_index(self, data, buffer_size=50000):
        n = len(data)
        bs = buffer_size
        for i in range(0, n, bs):
            vectors = [np.reshape(t, (1, -1)) for t in data[i : i + bs]]
            vectors = np.concatenate(vectors, axis=0)
            self.index.add(vectors)
            logger.info(f"data hnsw indexed {self.index.ntotal}")
        logger.info(f"Total data hnsw indexed {self.index.ntotal}")

    def add(self, array):
        if self.ann_search:
            self.buffer.append(array)
        else:
            self.index.add(array)
        
    def save(self, path):
        if self.device >= 0:
            self.index = faiss.index_gpu_to_cpu(self.index)
            self.device = -1

        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, dim, path):
        faiss_index = cls(dim)
        index = faiss.read_index(path)
        faiss_index.index = index
        return faiss_index

    def to_cuda(self, device):
        self.device = device
        if self.device < 0:
            return
        
        faiss_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(faiss_res, self.device, self.index)
        self.index = index
        
    def search(self, queries, topk):
        return self.index.search(queries, topk)
        

class IndexingDataset(torch.utils.data.Dataset):
    def __init__(self, collections, batch_size, pad_token_id=0):
        self.collections = collections
        
        # set dummy dataset
        if len(collections) % batch_size != 0:
            i = len(collections) // batch_size
            self.length = (i + 1) * batch_size
            self.dummy_data = [0] * 384  # max_passage_length
        else:
            self.length = len(collections)
        self.pad_token_id = pad_token_id
        
    def __getitem__(self, idx):
        # dummy dataset
        if idx >= len(self.collections):
            x = self.dummy_data
            idx = -1
            return x, idx

        x = self.collections.get_data(idx)
        return x, idx

    def __len__(self):
        return self.length
    
    def collate_fn(self, batch):
        input_ids, ids = list(zip(*batch))
        input_ids = torch.LongTensor(input_ids)
        input_masks = input_ids.ne(self.pad_token_id)
        ids = list(ids)
        return ids, input_ids, input_masks
    
    
class DenseIndexer:
    def __init__(self,
                 input_dir,
                 dim=768,
                 batch_size=128,
                 max_buffer_size=592000,
                 logger=None):
        self.input_dir = input_dir
        self.dim = dim
        self.batch_size = batch_size
        self.index = None
        self.max_buffer_size = max_buffer_size
        self.logger = logger
        faiss.omp_set_num_threads(16)

    def set_collections(self):
        self.collections = DocumentCollection(f"{self.input_dir}/data.h5")
        
    def load_index(self, index_path):
        index = FaissIndex.load(self.dim, index_path)
        self.index = index
        if self.logger:
            dist_print(f"Loading precomputed index..! {len(self.index)} number of indices", self.logger)
        
    def save_index(self, save_path):
        if self.index:
            self.index.save(save_path)
        
    def passage_inference(self, ctx_encoder, output_path=None, rank=0, world_size=1, ann_search=False):
        device = ctx_encoder.device
        dataset = IndexingDataset(self.collections, self.batch_size * world_size)
        index = FaissIndex(self.dim, ann_search=ann_search)

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self.batch_size * world_size,
                                             shuffle=False,
                                             collate_fn=dataset.collate_fn)
        all_ids = []
        all_vectors = []
        if self.logger:
            dist_print(f"Start inference! {self.max_buffer_size}", self.logger)
        
        for idx, batch in enumerate(loader):
            all_local_ids, all_input_ids, all_input_masks = batch
            
            local_ids = data_sharding(all_local_ids, world_size, rank)
            input_ids = data_sharding(all_input_ids, world_size, rank)
            input_masks = data_sharding(all_input_masks, world_size, rank)

            with torch.no_grad():
                outputs = ctx_encoder(input_ids.to(device),
                                      attention_mask=input_masks.to(device))
                local_ctx_vectors = outputs.pooler_output

            global_ctx_vectors, global_ids = all_gather_items([local_ctx_vectors, local_ids],
                                                              world_size,
                                                              rank,
                                                              self.max_buffer_size
                                                             )
            
            if is_main():
                global_ctx_vectors = global_ctx_vectors.cpu().detach().numpy()
                if -1 in global_ids:
                    global_ids = [i for i in global_ids if i >= 0]
                    global_ctx_vectors = global_ctx_vectors[:len(global_ids)]
                index.add(global_ctx_vectors)
                all_ids.extend(global_ids)

            if self.logger and idx % 100 == 0:
                dist_print(f"Write passage index ... [{len(all_ids)}/{len(self.collections)}]", self.logger)

            if dist.is_initialized():
                dist.barrier(device_ids=[rank])

        if is_main() and output_path:
            if index.ann_search:
                index.buffer = np.vstack(index.buffer)
                index.construct_hnsw_index(index.buffer)
            
            index.save(output_path)
            tmp = "/".join(output_path.split("/")[:-1])
            json.dump(all_ids, open(f"{tmp}/output.json", "w"), indent=2)

        if dist.is_initialized():
            dist.barrier(device_ids=[rank])

        if self.logger:
            dist_print(f"Indexing {len(index)} done.", self.logger)
        self.index = index
    
    def retrieve(self,
                 inputs,
                 qids,
                 top_k=100):
        if not self.index:
            return {}

        D, I = self.index.search(inputs, top_k)
        score_dict = {}
        index_dict = {}
        for scores, pids, qid in zip(D, I, qids):
            score_dict[qid] = {self.collections.get_pid(pid): float(score) for score, pid in zip(scores, pids)}
            index_dict[qid] = pids.tolist()

        return score_dict, index_dict


class SparseIndexer:
    
    def __init__(self, index_path=None):
        self.index_path = index_path
        
    def set_retriever(self, k_1=0.82, b=0.68):
        self.searcher = SimpleSearcher(self.index_path)
        self.searcher.set_bm25(k_1, b)
        
    def retrieve(self, question, num_passages, return_raw=False):
        hits = self.searcher.search(question, k=num_passages)
        
        if return_raw:
            return [json.loads(hit.raw) for hit in hits]
        
        return [{"id": hit.docid, "score": hit.score, "text": json.loads(hit.raw)["contents"]} for hit in hits]

    def write_jsonl(self, output_dir, passage_file):
        for idx, line in enumerate(open(passage_file, "r", encoding="utf-8")):
            example = json.loads(line)  # TODO: title?
            text = example['text']
            example_id = example['id']
            dic = {
                'id': example_id,
                'contents': text,
            }
            
            with open(f"{output_path}/data.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(dic) + '\n')
    
    @staticmethod
    def make_sparse_index(input_dir, save_path, n_threads=4):
        cmd = f"python3 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads {n_threads} -input {input_dir} -index {save_path} -storePositions -storeDocvectors -storeRaw"    
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        for line in iter(process.stdout.readline, ''):
            if type(line) == bytes:
                line = line.decode('utf-8')
            if not line:
                break

            try:
                sys.stdout.write(line)
            except Exception as e:
                print(e)
                break
