import os
import json
from .base import RetrievalInstance, EOT_TOKEN


class OrConvQAProcessor:
    
    def __init__(self,
                 data_dir,
                 q_tokenizer,
                 ctx_tokenizer,
                 max_query_length=128,
                 max_passage_length=384,
                 retain_first_question=False,
                 use_only_queries=True,
                 use_rewrite_only=False,
                 verbose=False,
                 logger=None):
        self.data_dir = data_dir
        self.q_tokenizer = q_tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.retain_first_question = retain_first_question
        self.use_only_queries = use_only_queries
        self.use_rewrite_only = use_rewrite_only
        self.verbose = verbose
        self.logger = logger
        self.answer_map = {}

    def read_examples(self, file_name, skip_no_truth_passages=False):
        examples = []
        with open(os.path.join(self.data_dir, file_name), "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                example = self.preprocess(line)
                examples.append(example)
                
                if self.verbose and self.logger is not None and len(examples) % 100 == 0:
                    self.logger.info(f"{len(examples)} examples done")
        return examples

    def preprocess(self, line):
        guid = line["qid"]
        first_query = None
        history = []
        for idx, q in enumerate(line["history"]):
            text = q["question"]
            if idx == 0:
                first_query = text + f" {EOT_TOKEN}"
            else:
                history.append(text)
                
        if not self.retain_first_question and first_query:
            history.insert(0, first_query)
            first_query = None
        
        did, _ = guid.split("#")
        if self.use_rewrite_only:
            # for mannual
            query_tokens = self.q_tokenizer.tokenize(line["rewrite"])
            max_available_length = self.max_query_length - 2
            if len(query_tokens) > max_available_length:
                gap = len(query_tokens) - max_available_length
                query_tokens = query_tokens[:gap]
            query_tokens = [self.q_tokenizer.cls_token] + query_tokens + [self.q_tokenizer.sep_token]
            inputs = self.q_tokenizer.convert_tokens_to_ids(query_tokens)
        else:
            previous_answer = None
            if self.answer_map.get(did):
                previous_answer = self.answer_map[did]

            if previous_answer and not self.use_only_queries:
                history.append(previous_answer)

            history.append(line["question"])
            history = f" {EOT_TOKEN} ".join(history)

            if first_query:
                first_query_tokens = self.q_tokenizer.tokenize(first_query)
            else:
                first_query_tokens = []

            max_available_length = self.max_query_length - 2 - len(first_query_tokens)
            query_tokens = self.q_tokenizer.tokenize(history)

            if len(query_tokens) > max_available_length:
                gap = len(query_tokens) - max_available_length
                query_tokens = query_tokens[gap:]

            query_tokens = [self.q_tokenizer.cls_token] + first_query_tokens + query_tokens + [self.q_tokenizer.sep_token]
            inputs = self.q_tokenizer.convert_tokens_to_ids(query_tokens)

        mask = [1] * len(inputs)

        while len(inputs) < self.max_query_length:
            inputs.append(self.q_tokenizer.pad_token_id)
            mask.append(0)

        title = line.get("title", "")
        title = [title] * len(line["evidences"])
        candidate = self.ctx_tokenizer(title,
                                       text_pair=line["evidences"],
                                       padding="max_length",
                                       truncation=True,
                                       max_length=self.max_passage_length)
        
        label = line["retrieval_labels"].index(1) if 1 in line["retrieval_labels"] else 0
        
        if line.get("rewrite"):
            rewrite_id = self.q_tokenizer.encode(line["rewrite"])
        else:
            rewrite_id = None
        
        instance = RetrievalInstance(guid,
                                     "",
                                     inputs,
                                     mask,
                                     candidate["input_ids"],
                                     candidate["attention_mask"],
                                     label,
                                     hard_negative_idx=-1,
                                     rewrite_id=rewrite_id)
        self.answer_map[did] = line["answer"]["text"]

        return instance
