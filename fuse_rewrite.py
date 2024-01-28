import json
import os
import jsonlines
import argparse


def main(args):
    split = args.rw_file.split("_")[0]
    lines = []
    with jsonlines.open(os.path.join(args.root_path, args.rw_file)) as reader:
        for line in reader:
            if "Transformer_rewrite" in line:
                del line['Transformer_rewrite']
            lines.append(line)
#     lines = json.load(open(os.path.join(args.root_path, args.rw_file), "r"))
    print(f'Number of turns in {split}: {len(lines)}')
    
    # read qrels
    if split == "test":
        qrels = json.load(open(os.path.join(args.root_path, f"qrels_{split}.txt"), "r"))
    
    new_lines = []
    for line in lines:
        conv_id = f"{line['Conversation_no']}_{line['Turn_no']}"
        no_rels = False
        if split == "test":
            if list(qrels[conv_id].keys())[0] == '':
                no_rels = True
        if no_rels:
            continue
        if line['Turn_no'] > 1:
            line[args.rw_type] = line[args.rw_type]['choices'][0]['message']['content']
        new_lines.append(line)
        
    print(f'Number of left turns in {split}: {len(new_lines)}')
    
    out_file = args.rw_file[args.rw_file.find("chatgpt_"):].strip(".jsonl").lstrip("chatgpt_")
    if args.suffix is not None:
        out_file = out_file + "_" + args.suffix
    with open(os.path.join(args.root_path, f'{split}_fused_{out_file}.json'), 'w') as out:
        json.dump(new_lines, out, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=None)
    parser.add_argument('--rw_file', type=str, default=None)
    parser.add_argument('--rw_type', type=str, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    args = parser.parse_args()

    main(args)