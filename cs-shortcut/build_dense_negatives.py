import os
import sys
import json
import torch
import argparse
from transformers import DPRQuestionEncoder, DPRContextEncoder, AutoTokenizer
from utils.indexing_utils import DenseIndexer, data_sharding, DocumentCollection

from data.base import load_processed_data
from dpr import DPRForPretraining
from utils import retrieval_utils, indexing_utils, set_seed, get_logger
from train_retriever import evaluation


logger = get_logger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_initial_checkpoint(output_path):
    q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model = DPRForPretraining(q_encoder, ctx_encoder)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--output_path', type=str, default="outputs")
    parser.add_argument('--task', type=str, default='qrecc')
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--index_batch_size', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--iteration', type=int, default=1)
    args = parser.parse_args()

    set_seed(args.random_seed)
    args.qrel_path = os.path.join(args.data_path, args.task, 'qrels.txt')
    if not os.path.exists(args.model_name_or_path):
        make_initial_checkpoint(args.model_name_or_path)

    q_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    ctx_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = DPRForPretraining.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()
    
    examples = load_processed_data(os.path.join(args.data_path, args.task, f"{args.split}.json"))
    eval_data = retrieval_utils.RetrievalDataset(examples)
    eval_loader = torch.utils.data.DataLoader(eval_data,
                                              batch_size=args.index_batch_size,
                                              shuffle=False,
                                              collate_fn=eval_data.collate_fn)

    indexer = DenseIndexer(os.path.join(args.data_path, args.task, "test_collections"),
                           batch_size=args.index_batch_size, logger=logger)
    indexer.set_collections()

    if os.path.exists(f"{args.model_name_or_path}/index_test.faiss"):
        indexer.load_index(f"{args.model_name_or_path}/index_test.faiss")
        logger.info(f"Index loading success!!")
    else:
        indexer.passage_inference(model.ctx_encoder,
                                  f"{args.model_name_or_path}/index_test.faiss",
                                  rank=0,
                                  world_size=1)

    eval_result, scores, indices = evaluation(eval_loader,
                                              model,
                                              indexer,
                                              args.qrel_path,
                                              args.top_k,
                                              world_size=1,
                                              rank=0
                                             )
    
    if not args.split.startswith("train"):
        logger.info(f"Test score: {eval_result}")

    json.dump(
        eval_result,
        open(os.path.join(args.model_name_or_path, f"{args.split}_eval_result.json"), "w"),
        indent=4
    )
    json.dump(
        scores,
        open(os.path.join(args.model_name_or_path, f"{args.split}_eval_scores.json"), "w"),
        indent=4
    )
    json.dump(
        indices,
        open(os.path.join(args.model_name_or_path, f"{args.split}_eval_indices.json"), "w"),
        indent=4
    )

    logger.info(f"Hard Negatives Mining is done")
    with open(os.path.join(args.model_name_or_path, f"{args.split}_negs.json"), "w") as f:
        for example in examples:
            negative_ids = indices[example.guid]
            negative_scores = scores[example.guid]
            example.hard_negative_ids = negative_ids
            example.hard_negative_scores = negative_scores
            f.write(json.dumps(example.to_dict()) + "\n")
