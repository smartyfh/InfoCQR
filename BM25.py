import os
import json
import argparse
from tqdm import tqdm

from shared_utils.indexing_utils import SparseIndexer, DocumentCollection
from shared_utils import get_logger


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
        elif read_by == "rewrite":
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="qrecc")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--read_by', type=str, default="all")
    parser.add_argument('--raw_data_path', type=str, default=None)
    parser.add_argument('--preprocessed_data_path', type=str, default=None)
    parser.add_argument('--pyserini_index_path', type=str, default=None)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=100)
    args = parser.parse_args()
    
    os.makedirs(args.preprocessed_data_path, exist_ok=True)

    k_1 = 0.82
    b = 0.68

    indexer = SparseIndexer(args.pyserini_index_path)
    indexer.set_retriever(k_1, b)

    qrels = json.load(open(os.path.join("datasets/qrecc", f"qrels_{args.split}.txt"), "r"))

    data = json.load(open(f"{args.raw_data_path}/{args.task}/{args.data_file}", "r", encoding="utf-8"))
    if args.read_by == "model":
        raw_examples = read_qrecc_data_model_pred(data)
    else:
        raw_examples = read_qrecc_data(data, args.read_by)
    print(f'Total number of queries: {len(raw_examples)}')

    scores = {}
    for idx, line in enumerate(raw_examples):
        qid, q = line

        no_rels = False
        if args.split == "test" or args.split == "dev":
            if list(qrels[qid].keys())[0] == '':
                no_rels = True
        if no_rels:
            continue
        
        if not q:
            scores[qid] = {}
            continue

        retrieved_passages = indexer.retrieve(q, args.top_k)
        score = {}
        for passage in retrieved_passages:
            score[passage["id"]] = passage["score"]
        scores[qid] = score
        logger.info(f"{idx}/{len(raw_examples)}")
    
    out_sfx = args.data_file.lstrip(args.split+"_").strip(".json")
    json.dump(
        scores,
        open(os.path.join(args.preprocessed_data_path, f"{args.split}_{args.read_by}_{out_sfx}_bm25.json"), "w"),
        indent=4
    )
    
