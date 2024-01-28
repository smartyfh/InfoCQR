import os
import json
import argparse
import logging
from tqdm import tqdm, trange

from utils.indexing_utils import DenseIndexer, DocumentCollection
from utils import get_logger
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


def read_qrecc_data(dataset, read_by="all", is_test=False):
    examples = []
    for data in tqdm(dataset):
        guid = f"{data['Conversation_no']}_{data['Turn_no']}"
        context = data['NewContext']
        assert len(context) % 2 == 0

        target_question = data["Question"]
        
        if read_by == "all":
            x = context + [target_question]
            x = " ".join(x)
        elif read_by == "all_without_this":
            x = context
            x = " ".join(x)
        elif read_by == "Truth_rewrite":
            x = data["Truth_rewrite"]
        elif read_by == "GPT_rewrite":
            x = data["GPT_rewrite"]
        elif read_by == "Editor_rewrite":
            x = data["Editor_rewrite"]
        elif read_by == "original":
            x = data["Question"]
        elif read_by == "this_answer":
            x = [data["Question"], data["Truth_answer"]]
            x = " ".join(x)
        elif read_by == "this_answer_only":
            x = data["Truth_answer"]
        else:
            raise Exception("Unsupported option!")

        examples.append([guid, x])
        
        if is_test:
            logger.info(f"{guid}: {x}")
            if len(examples) == 10:
                break
        
    return examples


def read_qrecc_data_model_pred(dataset):
    examples = []
    for did in tqdm(dataset.keys()):
        new_did = "_".join(did.split("_")[-2:])
        examples.append([new_did, dataset[did]['pred']])
  
    return examples


def merge_scores(scores_list, topk):
    results = {}
    for rr in scores_list:
        for k, v in rr.items():
            if k not in results:
                results[k] = list(v.items())
            else:
                results[k].extend(list(v.items()))
                
    new_results = {}
    for k, v in results.items():
        new_results[k] = {}
        vv = sorted(v, key=lambda x: -x[1])
        for i in range(topk):
            pid, ss = vv[i]
            new_results[k][pid] = ss
            
    return new_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="qrecc")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--read_by', type=str, default="Truth_rewrite")
    parser.add_argument('--raw_data_path', type=str, default="/home/ye/CQR/datasets")
    parser.add_argument('--preprocessed_data_path', type=str, default="/home/ye/CQR/outputs/DPR")
    parser.add_argument('--dense_index_path', type=str, default="/home/ye/CQR/datasets/preprocessed/qrecc/dense_index")
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--num_splits', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=100)
    args = parser.parse_args()
    
    if not os.path.exists(args.preprocessed_data_path):
        os.makedirs(args.preprocessed_data_path, exist_ok=True)
        
    fileHandler = logging.FileHandler(f"{args.preprocessed_data_path}/log.out", "a")
    formatter = logging.Formatter('%(asctime)s > %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info("logging start!")
    
    # read data
    data = json.load(open(f"{args.raw_data_path}/{args.task}/{args.data_file}", "r", encoding="utf-8"))
    if args.read_by == "model":
        raw_examples = read_qrecc_data_model_pred(data)
    else:
        raw_examples = read_qrecc_data(data, args.read_by)
    print(f'Total number of queries: {len(raw_examples)}')
    
    qids = []
    queries = []
    for idx, line in enumerate(raw_examples):
        qid, q = line
        if q:
            qids.append(qid)
            queries.append(q)
    print(f'Number of valid queries: {len(queries)}')
    
    # query encoder
    model = SentenceTransformer('sentence-transformers/gtr-t5-base')
    model.max_seq_length = 384

    # query embeddings
    embeddings = model.encode(queries, 
                              batch_size=256, 
                              show_progress_bar=True) 
    
    all_scores_list = []
    out_sfx = args.data_file.lstrip(args.split+"_").strip(".json")
    for spt in range(args.num_splits):
        all_scores = {}
        for idx, line in enumerate(raw_examples):
            qid, q = line
            if not q:
                all_scores[qid] = {}
                continue
                
        # load passage ids
        pids_path = os.path.join(args.dense_index_path, f"doc_ids_{spt}.json")
        pids = json.load(open(pids_path, "r"))
        logger.info(f"Load {len(pids)} pids from {pids_path}")
            
        # load faiss index
        index_path = os.path.join(args.dense_index_path, f"index_test_{spt}.faiss")
        logger.info(f"Load index from {index_path}")
        indexer = DenseIndexer(dim=768,logger=logger)
        indexer.load_index(index_path)
        logger.info(f"Index loading success!!")
    
        scores = indexer.retrieve(embeddings, qids, pids)
    
        all_scores.update(scores)
        
        logger.info(f"Dense search finished")
        
        all_scores_list.append(all_scores)
    
        json.dump(
            all_scores,
            open(os.path.join(args.preprocessed_data_path, f"{args.split}_{args.read_by}_{out_sfx}_dpr_{spt}.json"), "w"),
            indent=4
        )
    
    merged_results = merge_scores(all_scores_list, 100)
    json.dump(
            merged_results,
            open(os.path.join(args.preprocessed_data_path, f"{args.split}_{args.read_by}_{out_sfx}_dpr.json"), "w"),
            indent=4
        )
