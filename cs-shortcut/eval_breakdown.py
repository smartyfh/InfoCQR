import os
import json
import argparse
import numpy as np
import pytrec_eval


def question_breakdown(result_path, qrel_path, subdomain_path):
    all_result = json.load(open(result_path, "r"))
    qrels = json.load(open(qrel_path, "r"))
    subdomain = json.load(open(subdomain_path, "r"))
    subdomain["all"] = list(all_result.keys())
    
    print(result_path + "\n")
    for source, pids in subdomain.items():
        sqrels = dict(filter(lambda x: x[0] in pids, qrels.items()))
        sqrels = dict(filter(lambda x: x[1] != {"": 1}, sqrels.items())) # QReCC: filtering missings
        sresults = dict(filter(lambda x: x[0] in pids, all_result.items()))

        evaluator = pytrec_eval.RelevanceEvaluator(
                sqrels, {"recip_rank", "recall"})
        metrics = evaluator.evaluate(sresults)
        mrr_list = [v["recip_rank"] for v in metrics.values()]
        recall_5_list = [v["recall_5"] for v in metrics.values()]
        recall_10_list = [v["recall_10"] for v in metrics.values()]
        recall_20_list = [v["recall_20"] for v in metrics.values()]
        recall_100_list = [v["recall_100"] for v in metrics.values()]
        eval_metrics = {
            "MRR": np.average(mrr_list),
            "Recall@5": np.average(recall_5_list),
            "Recall@10": np.average(recall_10_list),
            "Recall@20": np.average(recall_20_list),
            "Recall@100": np.average(recall_100_list)
        }
        print(source, len(sqrels))
        print(eval_metrics)
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="qrecc")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--result_file_path", type=str, default=None, required=True)
    args = parser.parse_args()

    qrel_path = os.path.join(args.data_path, args.task, "qrels.txt")
    qtype_path = os.path.join(args.data_path, args.task, "test_question_types.json")
    question_breakdown(args.result_file_path, qrel_path, qtype_path)
