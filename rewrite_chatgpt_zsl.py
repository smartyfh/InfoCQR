import json
import os
import openai
import jsonlines
from tqdm import tqdm
import time

model_name = "gpt-3.5-turbo"
openai.api_key = 'your key'


Instruction = "Given a question and its context, decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context."


def generate_rewrite(line):
    curr_ctx = []
    for idx, sent in enumerate(line['NewContext']):
        if idx % 2 == 0:
            curr_ctx.append(f"Q: {sent}")
        else:
            curr_ctx.append(f"A: {sent}")
    curr_ctx = " ".join(curr_ctx)
    curr_ctx = f"[{curr_ctx}]"
    prompt = f"{Instruction}\n\nContext: {curr_ctx}\nQuestion: {line['Question']}\nRewrite: "
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    retries = 5
    delay = 1
    while retries > 0:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=0, # greedy decoding
                max_tokens=2560,
#                 top_p=0.95,
                n=1,
            )
            return response
        except:
            pass
        retries -= 1
        time.sleep(delay)
        delay *= 2
    return ""


if __name__ == "__main__":
    split = 'test'
    root = 'datasets/qrecc/'
    lines = json.load(open(os.path.join(root, f'{split}-modified.json'), "r", encoding="utf-8"))
    
    if split == "test" or split == "dev":
        qrels = json.load(open(os.path.join(root, f"qrels_{split}.txt"), "r"))
    
    with jsonlines.open(os.path.join(root, f'{split}_chatgpt_ZSL.jsonl'), mode='a') as writer:
        for line in tqdm(lines):
            conv_id = f"{line['Conversation_no']}_{line['Turn_no']}"
            no_rels = False
            if split == "test" or split == "dev":
                if list(qrels[conv_id].keys())[0] == '':
                    no_rels = True
            if no_rels:
                continue
            
            if len(line['NewContext']) == 0:
                line['GPT_rewrite'] = line['Truth_rewrite']
            else:
                line['GPT_rewrite'] = generate_rewrite(line)
#             del line['Context']
            writer.write(line)
    
