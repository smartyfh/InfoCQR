import json
import os
import openai
import jsonlines
from tqdm import tqdm
import time

model_name = "gpt-3.5-turbo"
openai.api_key = 'your key'


Instruction = "Given a question and its context and a rewrite that decontextualizes the question, edit the rewrite to create a revised version that fully addresses coreferences and omissions in the question without changing the original meaning of the question but providing more information. The new rewrite should not duplicate any previously asked questions in the context. If there is no need to edit the rewrite, return the rewrite as-is."

e1 = "Context: [Q: When was Born to Fly released? A: Sara Evans's third studio album, Born to Fly, was released on October 10, 2000.]\nQuestion: Was Born to Fly well received by critics?\nRewrite: Was Born to Fly well received by critics?\nEdit: Was Born to Fly well received by critics?"

e2 = "Context: [Q: When was Keith Carradine born? A: Keith Ian Carradine was born August 8, 1949. Q: Is he married? A: Keith Carradine married Sandra Will on February 6, 1982.]\nQuestion: Do they have any children?\nRewrite: Does Keith Carradine have any children?\nEdit: Do Keith Carradine and Sandra Will have any children?"

e3 = "Context: [Q: Who proposed that atoms are the basic units of matter? A: John Dalton proposed that each chemical element is composed of atoms of a single, unique type, and they can combine to form more complex structures called chemical compounds.]\nQuestion: How did the proposal come about?\nRewrite: How did John Dalton's proposal come about?\nEdit: How did John Dalton's proposal that each chemical element is composed of atoms of a single unique type, and they can combine to form more complex structures called chemical compounds come about?"

e4 = "Context: [Q: What is it called when two liquids separate? A: Decantation is a process for the separation of mixtures of immiscible liquids or of a liquid and a solid mixture such as a suspension. Q: How does the separation occur? A: The layer closer to the top of the container-the less dense of the two liquids, or the liquid from which the precipitate or sediment has settled out-is poured off.]\nQuestion: Then what happens?\nRewrite: Then what happens after the layer closer to the top of the container is poured off?\nEdit: Then what happens after the layer closer to the top of the container is poured off with decantation?"


def generate_rewrite(line):
    curr_ctx = []
    for idx, sent in enumerate(line['NewContext']):
        if idx % 2 == 0:
            curr_ctx.append(f"Q: {sent}")
        else:
            curr_ctx.append(f"A: {sent}")
    curr_ctx = " ".join(curr_ctx)
    curr_ctx = f"[{curr_ctx}]"
    prompt = f"{Instruction}\n\n{e1}\n\n{e2}\n\n{e3}\n\n{e4}\n\nContext: {curr_ctx}\nQuestion: {line['Question']}\nRewrite: {line['T5_rewrite']}\nEdit: "
    
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
    T5_rewrites = json.load(open("T5QR/predictions/_lr1e-05_bs32_ep10.0_acc2_gl64_Truth_rewrite_new-pred.json", "r", encoding="utf-8"))
    
    with jsonlines.open(os.path.join(root, f'{split}_chatgpt_editor.jsonl'), mode='a') as writer:
        for line in tqdm(lines[0:]):
            conv_id = f"{line['Conversation_source']}_{line['Conversation_no']}_{line['Turn_no']}"
            line['T5_rewrite'] = T5_rewrites[conv_id]["pred"]
            if len(line['NewContext']) == 0:
                line['Editor_rewrite'] = line['Truth_rewrite']
            else:
                line['Editor_rewrite'] = generate_rewrite(line)
#             del line['Context']
            writer.write(line)
    
