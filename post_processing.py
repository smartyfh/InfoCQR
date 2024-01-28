import json
import os
import argparse


"""
Some responses are N/A. 
E.g., N/A - The rewrite does not need to be edited as it is already a complete question that does not require any additional information from the context.
For these responses, we change them back to the original question or model rewrite
"""

"""
Some responses are not questions.
E.g., There is no clear question in the context that can be decontextualized and rewritten to retain its original meaning.

or

No edit needed, the rewrite already fully addresses coreferences and omissions in the question.
"""



def main(args):
    split = args.rw_file.split("_")[0]
    
    lines = json.load(open(os.path.join(args.root_path, args.rw_file), "r"))
    print(f'Number of turns in {split}: {len(lines)}')
    
    num_NA = 0
    num_NQ = 0
    num_NE = 0
    for line in lines:
        if line["Turn_no"] > 1:
            if "N/A" in line[args.rw_type]:
                num_NA += 1
                if "Editor" in args.rw_type:
                    line[args.rw_type] = line[args.init_rw_type]
                else:
                    line[args.rw_type] = line["Question"]
            elif line[args.rw_type][-1] != "?":
                num_NQ += 1
                if "Editor" in args.rw_type:
                    line[args.rw_type] = line[args.init_rw_type]
                else:
                    line[args.rw_type] = line["Question"]
            elif "no edit" in line[args.rw_type].lower() or "no need to edit" in line[args.rw_type].lower():
                num_NE += 1
                if "Editor" in args.rw_type:
                    line[args.rw_type] = line[args.init_rw_type]
                else:
                    line[args.rw_type] = line["Question"]
    
    print(f'num_NA: {num_NA}; num_NQ: {num_NQ}; num_NE: {num_NE}')
    
    out_file = args.rw_file.strip(".json")
    with open(os.path.join(args.root_path, f'{out_file}_post.json'), 'w') as out:
        json.dump(lines, out, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=None)
    parser.add_argument('--rw_file', type=str, default=None)
    parser.add_argument('--rw_type', type=str, default=None)
    parser.add_argument('--init_rw_type', type=str, default="GPT_rewrite")
    args = parser.parse_args()

    main(args)