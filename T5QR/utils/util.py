import os
import numpy as np
import evaluate

class Metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def compute_metrics(self, eval_preds):
        predictions, labels = eval_preds
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        clf_metrics = evaluate.combine(['bleu', 'rouge', 'meteor'])
        results = clf_metrics.compute(predictions=decoded_preds, references=decoded_labels)
        if "precisions" in results: # a list
            pp = results["precisions"]
            del results["precisions"]
            results["bleu_prec_1"] = pp[0]
            results["bleu_prec_2"] = pp[1]
            results["bleu_prec_3"] = pp[2]
            results["bleu_prec_4"] = pp[3]
            
        return results

def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it
    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir
    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(args.output_dir)
        and len(os.listdir(args.output_dir)) > expected_items
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )
