"""
https://github.com/scai-conf/SCAI-QReCC-21/blob/main/code/evaluation-script/scai-qrecc21-evaluator.py
"""

import json
import argparse
import pytrec_eval


def get_turn_id(turn):
    return "%d_%d" % (turn["Conversation_no"], turn["Turn_no"])


def get_retrieval_run(run):
    retrieval_run = {}
    for turn in run:
        turn_id = get_turn_id(turn)
        if "Model_passages" in turn:
            if "" in turn["Model_passages"]:
                sys.exit("Invalid passage ID: '' for turn %s" % turn_id)
            retrieval_run[turn_id] = turn["Model_passages"]
    return retrieval_run

def get_retrieval_ground_truth(ground_truth, eval_missing_truth):
    retrieval_ground_truth = {}
    for turn in ground_truth:
        turn_id = get_turn_id(turn)
        if "Truth_passages" in turn and len(turn["Truth_passages"]) > 0:
            retrieval_ground_truth[turn_id] = {passage:1 for passage in turn["Truth_passages"]}
        elif eval_missing_truth: # paper version
            retrieval_ground_truth[turn_id] = {"":1}
    return retrieval_ground_truth


def evaluate_retrieval(ground_truth, run, eval_missing_truth):
    print("Evaluate: Passage Retrieval")
    result = {}
    retrieval_run = get_retrieval_run(run)
    retrieval_ground_truth_for_type = get_retrieval_ground_truth(ground_truth, eval_missing_truth)
    retrieval_run_for_type = {turn_id:passages for (turn_id, passages) in retrieval_run.items() if turn_id in retrieval_ground_truth_for_type}
    if retrieval_run_for_type: # at least one turn for this type => evaluate
        metric = pytrec_eval.RelevanceEvaluator(retrieval_ground_truth_for_type, {'recip_rank', 'recall'})
        metrics = metric.evaluate(retrieval_run_for_type)
        mrrs = [score["recip_rank"] for score in metrics.values()]
        recalls_100 = [v['recall_100'] for v in metrics.values()]
        recalls_10 = [v['recall_10'] for v in metrics.values()]
        average_mrr = sum(mrrs) / len(mrrs)
        average_recall_10 = sum(recalls_10) / len(recalls_10)
        average_recall_100 = sum(recalls_100) / len(recalls_100)
        result["MRR"] = average_mrr
        result["Recall@10"] = average_recall_10
        result["Recall@100"] = average_recall_100
        print("    used retrieved passages for %d questions" % len(retrieval_run_for_type))
    else:
        print("    skipped for no retrieved passages")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset_file', type=str, default=None)
    parser.add_argument('--ground_truth_file', type=str, default=None)
    args = parser.parse_args()

    run = json.load(open(args.input_dataset_file, "r"))
    ground_truth = json.load(open(args.ground_truth_file, "r"))
    result = evaluate_retrieval(ground_truth, run, True)
    print(result)
