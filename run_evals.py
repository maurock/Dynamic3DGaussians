"""Run multiple evaluations. Evaluations need to be on the same dataset."""
import eval
import argparse
import output
import os

def main(args):
    for exp_name in args.exp_names:
        output_seqs = os.listdir(
            os.path.join(
                os.path.dirname(output.__file__),
                exp_name
            )
        )
        for output_seq in output_seqs:
            evaluator = eval.Evaluator(args.dataset, exp_name, output_seq, args.save_eval_data)
            evaluator.run_evaluation()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_names", type=str, nargs='+', help="Path to the experiment directories inside output/, e.g. exp1")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name. Choose between 'ShinyBlender', 'GlossySynthetic'")
    parser.add_argument("--save_eval_data", default=False, action='store_true', help="Save evaluation data (True/False)")
    args = parser.parse_args()

    main(args)
    
