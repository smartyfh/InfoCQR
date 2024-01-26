import os
import json
import argparse
from copy import deepcopy


def main(args):
    for split in ['train', 'dev', 'test']:
        lines = json.load(open(os.path.join(args.root_path, f'{split}.json'), "r", encoding="utf-8"))
        print(f'Number of turns in {split}: {len(lines)}')
        
        last_question = None
        last_answer = None
        for idx, line in enumerate(lines):
            assert line['Turn_no'] >= 1
            if line['Turn_no'] == 1:
                # new conversation
                assert len(line['Context']) == 0
                line['Question'] = line['Truth_rewrite']
                last_question = line['Question']
                last_answer = line['Truth_answer']
                line['NewContext'] = []
            else:
                line['NewContext'] = deepcopy(lines[idx-1]['NewContext'])
                line['NewContext'].append(last_question)
                line['NewContext'].append(last_answer)
                last_question = line['Question']
                last_answer = line['Truth_answer']
            del line['Context']
            if "Transformer_rewrite" in line:
                del line['Transformer_rewrite']

        with open(os.path.join(args.root_path, split+"-modified.json"), 'w') as out:
            json.dump(lines, out, indent=2)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=None)
    args = parser.parse_args()

    main(args)
    
