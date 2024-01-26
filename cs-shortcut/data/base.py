import json
import re
from tqdm import tqdm
import dataclasses
from dataclasses import dataclass
from typing import List, Optional


EOT_TOKEN = "[SEP]"


def extract_split_from_file_name(file_name):
    check = re.search(r"(train|dev|test)", file_name)
    if check:
        return check.group()
    return ""


def load_processed_data(data_path):
    examples = []
    idx = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            example = RetrievalInstance.from_dict(json.loads(line.strip("\n")))
            examples.append(example)
            idx += 1
    return examples


@dataclass
class RetrievalInstance:
    guid: str
    context: List[str]
    input_id: List[int]
    input_mask: List[int]
    cand_ids: List[List[int]]
    cand_mask: List[List[int]]
    label: int
    hard_negative_idx: int
    label_id: Optional[List[int]] = None
    rewrite_id: Optional[List[int]] = None
    hard_negative_ids: Optional[List[int]] = None
    hard_negative_scores: Optional[List[int]] = None
    has_positive: Optional[bool] = True
    cons_hard_negative_ids: Optional[List[int]] = None
    switch_hard_negative_ids: Optional[List[int]] = None
    question_type: Optional[str] = None
        
    def to_dict(self):
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, dic):
        return cls(**dic)
