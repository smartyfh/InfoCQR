import json
from tqdm import tqdm
import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def len_preserved_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def len_preserved_space(matchobj):
        return ' ' * len(matchobj.group(0))

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', len_preserved_space, text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return remove_articles(remove_punc(lower(s)))


def split_with_span(s):
    if s.split() == []:
        return [], []
    else:
        return zip(*[(m.group(0), (m.start(), m.end()-1)) for m in re.finditer(r'\S+', s)])


def free_text_to_span(free_text, full_text):

    free_ls = len_preserved_normalize_answer(free_text).split()
    full_ls, full_span = split_with_span(len_preserved_normalize_answer(full_text))
    if full_ls == []:
        return full_text, 0, len(full_text)

    max_f1, best_index = 0.0, (0, len(full_ls)-1)
    free_cnt = Counter(free_ls)
    for i in range(len(full_ls)):
        full_cnt = Counter()
        for j in range(len(full_ls)):
            if i+j >= len(full_ls): break
            full_cnt[full_ls[i+j]] += 1

            common = free_cnt & full_cnt
            num_same = sum(common.values())
            if num_same == 0: continue

            precision = 1.0 * num_same / (j + 1)
            recall = 1.0 * num_same / len(free_ls)
            f1 = (2 * precision * recall) / (precision + recall)

            if max_f1 < f1:
                max_f1 = f1
                best_index = (i, j)

    assert(best_index is not None)
    (best_i, best_j) = best_index
    char_i, char_j = full_span[best_i][0], full_span[best_i+best_j][1]+1

    return full_text[char_i:char_j], char_i, char_j, max_f1


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


class ICTSpanMasker:
    
    def __init__(self, context, tokenizer, title=None, max_passage_length=384):
        self.tokenizer = tokenizer
        self.context = context
        self.title = title
        
        title_tokens = tokenizer.tokenize(f"{tokenizer.cls_token} {title} {tokenizer.sep_token}")
        self.title_tokens = title_tokens
        
        max_available_length = max_passage_length - len(self.title_tokens) - 1
        self.max_available_length = max_available_length
        
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in context:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset
        
        self.tok_to_orig_index = []
        self.orig_to_tok_index = []
        self.all_doc_tokens = []
        for (i, token) in enumerate(self.doc_tokens):
            self.orig_to_tok_index.append(len(self.all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                self.tok_to_orig_index.append(i)
                self.all_doc_tokens.append(sub_token)
                
        self.final_tokens = self.title_tokens + self.all_doc_tokens + [tokenizer.sep_token]
                
    def get_span_dict(self, texts, threshold=0.3):
        dic = {}
        for idx, text in enumerate(texts):
            span_text, start, end, f1 = free_text_to_span(text, self.context)
            if f1 < threshold:
                dic[idx] = (0, 0)
                continue
                
            start_position, end_position = self.get_span_in_context(start, span_text)
            if self.title_tokens:
                offset = len(self.title_tokens) - 1
                start_position += offset
                end_position += offset
            
            dic[idx] = (start_position, end_position)
        return dic

    def get_span_in_context(self, start_character, text):

        # Start and end positions only has a value during evaluation.
        if start_character is not None:
            start_position = self.char_to_word_offset[start_character]
            end_position = self.char_to_word_offset[
                min(start_character + len(text) - 1, len(self.char_to_word_offset) - 1)
            ]
            
        tok_start_position = self.orig_to_tok_index[start_position]
        if end_position < len(self.doc_tokens) - 1:
            tok_end_position = self.orig_to_tok_index[end_position + 1] - 1
        else:
            tok_end_position = len(self.all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            self.all_doc_tokens, tok_start_position, tok_end_position, self.tokenizer, text
        )
        
        return tok_start_position, tok_end_position
