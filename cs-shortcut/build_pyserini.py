import json
import os
import re
import shutil
import argparse
from pyserini.index import IndexReader
from utils.indexing_utils import SparseIndexer
from utils import get_logger


logger = get_logger(__name__)


def preprop(text):
    return re.sub(r'["]+', '"', text).strip('"')


def orconvqa_dump_preprocessing(dump_file_path, output_path):
    data = open(dump_file_path, "r", encoding="utf-8")
    for idx, line in enumerate(data):
        line = json.loads(line)
        title = line["title"]
        text = line.pop("text")
        contents = " [SEP] ".join([title, text])
        line["contents"] = contents
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    
        if idx % 100 == 0:
            logger.info(f"{idx} done")


def build_index(raw_jsonl_data_dir, pyserini_index_path, n_threads):
    SparseIndexer.make_sparse_index(input_dir=raw_jsonl_data_dir,
                                    save_path=pyserini_index_path,
                                    n_threads=n_threads)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--data_path', type=str, default="dataset")
    parser.add_argument('--output_path', type=str, default="preprocessed")
    parser.add_argument('--n_threads', type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    if args.task == "orconvqa":
        for dump_file in ["dev_blocks.txt", "all_blocks.txt"]:
            logger.info(f"Preprocessing collections of orconvqa: {dump_file}")
            orconvqa_dump_preprocessing(os.path.join(args.data_path, dump_file),
                                        os.path.join(args.data_path, f"tmp_{dump_file}")
                                       )
            shutil.move(os.path.join(args.data_path, f"tmp_{dump_file}"),
                        os.path.join(args.data_path, dump_file))

        if not os.path.exists(os.path.join(args.data_path, "raw_collections")):
            os.mkdir(os.path.join(args.data_path, "raw_collections"))
        
        src = os.path.join(args.data_path, "all_blocks.txt")
        trg = os.path.join(args.data_path, "raw_collections", "all_blocks.jsonl")
        shutil.move(src, trg)
        logger.info(f"Start building pyserini index of orconvqa")
        build_index(os.path.join(args.data_path, "raw_collections"),
                    os.path.join(args.output_path, "pyserini_index"),
                    args.n_threads)
        shutil.move(trg, src)
        
        # get title of gt context for train data
        index_reader = IndexReader(os.path.join(args.output_path, "pyserini_index"))
        qrels = json.load(open(os.path.join(args.data_path, "qrels.txt"), "r"))
        train_data = []
        for i, line in enumerate(open(os.path.join(args.data_path, "train.json"), "r")):
            line = json.loads(line)
            qid = line['qid']
            pid = list(qrels[qid].keys())[0]
            doc = index_reader.doc(pid)
            if not doc:
                raise Exception
                
            title = json.loads(doc.raw())["title"]
            line["title"] = title
            train_data.append(line)
        
        with open(os.path.join(args.data_path, "train.json"), "w") as f:
            for line in train_data:
                f.write(json.dumps(line) + "\n")
                
    elif args.task == "qrecc":
        logger.info(f"Start building pyserini index of {args.task}")
        build_index(os.path.join(args.data_path, "collection-paragraph"),
                    os.path.join(args.output_path, "pyserini_index"),
                    args.n_threads)
