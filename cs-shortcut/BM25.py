import os
import json
import argparse
from tqdm import tqdm

from utils.indexing_utils import SparseIndexer, DocumentCollection
from utils import get_logger


logger = get_logger(__name__)


def read_qrecc_data(dataset, read_by="all", is_test=False):
    context_map = {}
    prev_id = None
    examples = []
    for data in tqdm(dataset):
        guid = f"{data['Conversation_no']}_{data['Turn_no']}"
        did = data["Conversation_no"]
        context = context_map.get(did, [])
        assert len(context) % 2 == 0

        if not context:
            context_map[did] = []

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
        elif read_by == "previous_answer":
            if context:
                pa = context[-1]
            else:
                pa = ""
            x = [pa, data["Question"]]
            x = " ".join(x)
        elif read_by == "previous_answer_only":
            if context:
                pa = context[-1]
            else:
                pa = ""
            x = pa
        elif read_by == "this_answer":
            x = [data["Question"], data["Truth_answer"]]
            x = " ".join(x)
        elif read_by == "this_answer_only":
            x = data["Truth_answer"]
        else:
            raise Exception("Unsupported option!")

        examples.append([guid, x])
        
        context_map[did].append(data["Question"])
        context_map[did].append(data["Truth_answer"])
        
        if is_test:
            logger.info(f"{guid}: {x}")
            if len(examples) == 10:
                break
        
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--read_by', type=str, default="all")
    parser.add_argument('--raw_data_path', type=str, default=None)
    parser.add_argument('--preprocessed_data_path', type=str, default=None)
    parser.add_argument('--pyserini_index_path', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=100)
    args = parser.parse_args()

    if "qrecc" in args.task:
        k_1 = 0.82
        b = 0.68
    else:
        k_1 = 0.9
        b = 0.4

    indexer = SparseIndexer(args.pyserini_index_path)
    indexer.set_retriever(k_1, b)

    if args.task == "orconvqa":
        data = []
        for line in open(f"{args.raw_data_path}/{args.task}/{args.split}.json", "r", encoding="utf-8"):
            data.append(json.loads(line))
        raw_examples = read_orconvqa_data(data, args.read_by)
    else:
        data = json.load(open(f"{args.raw_data_path}/{args.task}/{args.split}_new_editor.json", "r", encoding="utf-8"))
        raw_examples = read_qrecc_data(data, args.read_by)

    scores = {}
    for idx, line in enumerate(raw_examples):
        qid, q = line
        if not q:
            scores[qid] = {}
            continue

        retrieved_passages = indexer.retrieve(q, args.top_k)
        score = {}
        for passage in retrieved_passages:
            score[passage["id"]] = passage["score"]
        scores[qid] = score
        logger.info(f"{idx}/{len(raw_examples)}")

    json.dump(
        scores,
        open(os.path.join(args.preprocessed_data_path, f"{args.split}_{args.read_by}_bm25_scores.json"), "w"),
        indent=4
    )
    