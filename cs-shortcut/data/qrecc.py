import os
import json
from .base import RetrievalInstance, EOT_TOKEN


class QReCCProcessor:
    
    def __init__(self,
                 data_dir,
                 q_tokenizer,
                 ctx_tokenizer,
                 index_reader,
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
        self.index_reader = index_reader
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.retain_first_question = retain_first_question
        self.use_only_queries = use_only_queries
        self.use_rewrite_only = use_rewrite_only
        self.verbose = verbose
        self.logger = logger
    
    def init_context_map(self):
        self.context_map = {}
    
    def read_examples(self, file_name, skip_no_truth_passages=False):
        self.init_context_map()
        split = file_name.replace(".json", "")
        examples = []
        with open(os.path.join(self.data_dir, file_name), "r", encoding="utf-8") as f:
            for line in json.load(f):

                example = self.preprocess(line, split)
                if not line["Truth_passages"]:
                    example.has_positive = False

                examples.append(example)
                
                if self.verbose and self.logger is not None and len(examples) % 100 == 0:
                    self.logger.info(f"{len(examples)} examples done")
                
        return examples

    def preprocess(self, line, split):
        did = line['Conversation_no']
        context = self.context_map.get(did, [])
        assert len(context) % 2 == 0

        if not context:
            self.context_map[did] = []
        
        if self.use_rewrite_only:
            query_tokens = self.q_tokenizer.tokenize(line['Truth_rewrite'])
            max_available_length = self.max_query_length - 2
            if len(query_tokens) > max_available_length:
                gap = len(query_tokens) - max_available_length
                query_tokens = query_tokens[:gap]
            query_tokens = [self.q_tokenizer.cls_token] + query_tokens + [self.q_tokenizer.sep_token]
            inputs = self.q_tokenizer.convert_tokens_to_ids(query_tokens)
        else:
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
            
#             if history:
#                 history = history[:-1] # without previous answer
            
            history.append(line['Question'])
            history = f" {EOT_TOKEN} ".join(history)
            
            self.context_map[did].append(line['Question'])
            self.context_map[did].append(line['Truth_answer'])

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

        guid = f"{line['Conversation_no']}_{line['Turn_no']}"
        
        evidences = []
        for did in line['Truth_passages']:
            doc = self.index_reader.doc(did)
            if doc:
                evidences.append(json.loads(doc.raw())['contents'])

        if not evidences:
            evidences = ['dummy']

        candidate = self.ctx_tokenizer(evidences,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_passage_length)
        label = 0  # TODO: what about multiple evidences?
        
        if line.get('Truth_rewrite'):
            query_tokens = self.q_tokenizer.tokenize(line['Truth_rewrite'])
            max_available_length = self.max_query_length - 2
            if len(query_tokens) > max_available_length:
                gap = len(query_tokens) - max_available_length
                query_tokens = query_tokens[:gap]
            query_tokens = [self.q_tokenizer.cls_token] + query_tokens + [self.q_tokenizer.sep_token]
            rewrite_id = self.q_tokenizer.convert_tokens_to_ids(query_tokens)
        else:
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
