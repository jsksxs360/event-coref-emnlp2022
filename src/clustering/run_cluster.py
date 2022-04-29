import os
import logging
from tqdm.auto import tqdm
import json
import subprocess
import re
import sys
sys.path.append('../../')
from src.clustering.arg import parse_args
from src.clustering.utils import create_golden_conll_file, get_pred_coref_results, create_pred_conll_file
from src.clustering.cluster import clustering

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Cluster")

COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)
BLANC_RESULTS_REGEX = re.compile(r".*BLANC: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)

def official_conll_eval(gold_path, predicted_path, metric, official_stdout=True):
    assert metric in ["muc", "bcub", "ceafe", "blanc"]
    cmd = ["../../reference-coreference-scorers/scorer.pl", metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        logger.error(stderr)

    if official_stdout:
        logger.info("Official result for {}".format(metric))
        logger.info(stdout)

    coref_results_match = re.match(
        BLANC_RESULTS_REGEX if metric == 'blanc' else COREF_RESULTS_REGEX, 
        stdout
    )
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}

if __name__ == '__main__':
    args = parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            f'Output directory ({args.output_dir}) already exists and is not empty.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    golden_conll_path = os.path.join(args.output_dir, args.golden_conll_filename)
    pred_conll_path = os.path.join(args.output_dir, args.pred_conll_filename)
    
    logger.info(f'creating golden conll file in {args.output_dir} ...')
    create_golden_conll_file(args.test_golden_filepath, golden_conll_path)
    # clustering
    # {doc_id: {'events': event_list, 'pred_labels': pred_coref_labels}}
    pred_coref_results = get_pred_coref_results(args.test_pred_filepath)
    cluster_dict = {} # {doc_id: [cluster_set_1, cluster_set_2, ...]}
    logger.info('clustering ...')
    for doc_id, pred_result in tqdm(pred_coref_results.items()):
        cluster_list = clustering(
            pred_result['events'], 
            pred_result['pred_labels'], 
            mode='rescore' if args.do_rescore else 'greedy', 
            rescore_reward=args.rescore_reward, 
            rescore_penalty=args.rescore_penalty
        )
        cluster_dict[doc_id] = cluster_list
    logger.info(f'saving predicted clusters in {args.output_dir} ...')
    create_pred_conll_file(cluster_dict, golden_conll_path, pred_conll_path)
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))
    # evaluate on the conll files
    if args.do_evaluate:
        results = {
            m: official_conll_eval(golden_conll_path, pred_conll_path, m, official_stdout=True) 
            for m in ("muc", "bcub", "ceafe", "blanc") 
        }
        results['avg_f1'] = sum([scores['f'] for scores in results.values()]) / len(results)
        logger.info(results)
        with open(os.path.join(args.output_dir, 'evaluate_results.json'), 'wt', encoding='utf-8') as f:
            f.write(json.dumps(results) + '\n')

