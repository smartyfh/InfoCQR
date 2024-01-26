import os
import json
from .mask import ICTSpanMasker
from .base import RetrievalInstance, EOT_TOKEN, extract_split_from_file_name


class TopioCQAProcessor:
    
    def __init__(self,
                 data_dir,
                 q_tokenizer,
                 ctx_tokenizer,
                 max_query_length=128,
                 max_passage_length=384,
                 retain_first_question=False,
                 use_only_queries=False,
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
        self.history_mask = True

    def read_examples(self, file_name, skip_no_truth_passages=False):
        examples = []
        with open(os.path.join(self.data_dir, file_name), "r", encoding="utf-8") as f:
            for idx, line in enumerate(json.load(f)):
                split = extract_split_from_file_name(line['dataset'])
                guid = f"{split}-{line['conv_id']}-{line['turn_id']}"
                line['guid'] = guid
                # TODO
                if skip_no_truth_passages and not line['positive_ctxs']:
                    continue

                example = self.preprocess(line)
                examples.append(example)
                
                if self.verbose and self.logger is not None and len(examples) % 100 == 0:
                    self.logger.info(f"{len(examples)} examples done")
                
        return examples
    
    def split_input(self, input_):
        context = input_.split(" [SEP] ")
        return context[:-1], context[-1]

    def preprocess(self, line):
        context, query = self.split_input(line["question"])
        first_query = None
        history = []
        for idx, text in enumerate(context):
            if self.use_only_queries and idx % 2 != 0:
                continue

            if idx == 0:
                first_query = text + f" {EOT_TOKEN}"
            else:
                history.append(text)

        if not self.retain_first_question and first_query:
            history.insert(0, first_query)
            first_query = None

        history.append(query)
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

        guid = line['guid']
        
        titles = []
        evidences = []
        for evidence in line['positive_ctxs']:
            titles.append(evidence['title'])
            evidences.append(evidence['text'])

        if not evidences:
            titles = ['dummy']
            evidences = ['dummy']

        candidate = self.ctx_tokenizer(titles,
                                       text_pair=evidences,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_passage_length)
        label = 0  # TODO: what about multiple evidences?
        rewrite_id = None 
        instance = RetrievalInstance(guid,
                                     "",
                                     inputs,
                                     mask,
                                     candidate['input_ids'],
                                     candidate['attention_mask'],
                                     label,
                                     hard_negative_idx=-1,
                                     rewrite_id=rewrite_id)
        return instance
