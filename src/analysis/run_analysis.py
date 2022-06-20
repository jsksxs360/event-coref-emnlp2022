from sklearn.metrics import classification_report
from collections import defaultdict
import sys
sys.path.append('../../')
from src.analysis.utils import get_event_pair_set

gold_coref_file = '../../data/test.json'
pred_coref_file = 'MaskTopicBN_M-multi-cosine.json'

def all_metrics(gold_coref_file, pred_coref_file):
    gold_coref_results, pred_coref_results = get_event_pair_set(gold_coref_file, pred_coref_file)
    all_event_pairs = [] # (gold_coref, pred_coref)
    for doc_id, gold_coref_result_dict in gold_coref_results.items():
        # {e_i_start-e_j_start: (coref, sent_dist, e_i_link_len, e_j_link_len)}
        gold_unrecognized_event_pairs, gold_recognized_event_pairs = (
            gold_coref_result_dict['unrecognized_event_pairs'], 
            gold_coref_result_dict['recognized_event_pairs']
        )
        pred_coref_result_dict = pred_coref_results[doc_id]
        # {e_i_start-e_j_start: (coref, sent_dist, e_i_link_len, e_j_link_len)}
        pred_recognized_event_pairs, pred_wrong_event_pairs = (
            pred_coref_result_dict['recognized_event_pairs'], 
            pred_coref_result_dict['wrong_event_pairs']
        )
        for pair_results in gold_unrecognized_event_pairs.values():
            all_event_pairs.append([str(pair_results[0]), '2'])
        for pair_id, pair_results in gold_recognized_event_pairs.items():
            all_event_pairs.append([str(pair_results[0]), str(pred_recognized_event_pairs[pair_id][0])])
        for pair_id, pair_results in pred_wrong_event_pairs.items():
            all_event_pairs.append(['0', str(pair_results[0])])
    y_true, y_pred = [res[0] for res in all_event_pairs], [res[1] for res in all_event_pairs]
    metrics = {'ALL': classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']}
    return metrics

def different_distance_metrics(gold_coref_file, pred_coref_file, adj_distance=3):
    gold_coref_results, pred_coref_results = get_event_pair_set(gold_coref_file, pred_coref_file)
    same_event_pairs, adj_event_pairs, far_event_pairs = [], [], []
    for doc_id, gold_coref_result_dict in gold_coref_results.items():
        # {e_i_start-e_j_start: (coref, sent_dist, e_i_link_len, e_j_link_len)}
        gold_unrecognized_event_pairs, gold_recognized_event_pairs = (
            gold_coref_result_dict['unrecognized_event_pairs'], 
            gold_coref_result_dict['recognized_event_pairs']
        )
        pred_coref_result_dict = pred_coref_results[doc_id]
        # {e_i_start-e_j_start: (coref, sent_dist, e_i_link_len, e_j_link_len)}
        pred_recognized_event_pairs, pred_wrong_event_pairs = (
            pred_coref_result_dict['recognized_event_pairs'], 
            pred_coref_result_dict['wrong_event_pairs']
        )
        for pair_results in gold_unrecognized_event_pairs.values():
            sent_dist = pair_results[1]
            pair_coref = [str(pair_results[0]), '2']
            if sent_dist == 0: # same sentence
                same_event_pairs.append(pair_coref)
            elif sent_dist < adj_distance: # adjacent sentence
                adj_event_pairs.append(pair_coref)
            else: # far sentence
                far_event_pairs.append(pair_coref)
        for pair_id, pair_results in gold_recognized_event_pairs.items():
            sent_dist = pair_results[1]
            pair_coref = [str(pair_results[0]), str(pred_recognized_event_pairs[pair_id][0])]
            if sent_dist == 0: # same sentence
                same_event_pairs.append(pair_coref)
            elif sent_dist < adj_distance: # adjacent sentence
                adj_event_pairs.append(pair_coref)
            else: # far sentence
                far_event_pairs.append(pair_coref)
        for pair_id, pair_results in pred_wrong_event_pairs.items():
            sent_dist = pair_results[1]
            pair_coref = ['0', str(pair_results[0])]
            if sent_dist == 0: # same sentence
                same_event_pairs.append(pair_coref)
            elif sent_dist < adj_distance: # adjacent sentence
                adj_event_pairs.append(pair_coref)
            else: # far sentence
                far_event_pairs.append(pair_coref)
    metrics = {}
    y_true, y_pred = [res[0] for res in same_event_pairs], [res[1] for res in same_event_pairs]
    metrics['SAME'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in adj_event_pairs], [res[1] for res in adj_event_pairs]
    metrics['ADJ'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    y_true, y_pred = [res[0] for res in far_event_pairs], [res[1] for res in far_event_pairs]
    metrics['FAR'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    return metrics
        
def main_link_metrics(gold_coref_file, pred_coref_file, main_link_length=5, mode='ge'):
    assert mode in ['g', 'ge', 'e', 'le', 'l']
    gold_coref_results, pred_coref_results = get_event_pair_set(gold_coref_file, pred_coref_file)
    main_link_event_pairs, singleton_event_pairs = [], defaultdict(list)
    for doc_id, gold_coref_result_dict in gold_coref_results.items():
        # {e_i_start-e_j_start: (coref, sent_dist, e_i_link_len, e_j_link_len)}
        gold_unrecognized_event_pairs, gold_recognized_event_pairs = (
            gold_coref_result_dict['unrecognized_event_pairs'], 
            gold_coref_result_dict['recognized_event_pairs']
        )
        pred_coref_result_dict = pred_coref_results[doc_id]
        # {e_i_start-e_j_start: (coref, sent_dist, e_i_link_len, e_j_link_len)}
        pred_recognized_event_pairs, pred_wrong_event_pairs = (
            pred_coref_result_dict['recognized_event_pairs'], 
            pred_coref_result_dict['wrong_event_pairs']
        )
        
        for pair_id, pair_results in gold_recognized_event_pairs.items():
            e_starts = pair_id.split('-')
            e_i_link_len, e_j_link_len = pair_results[2], pair_results[3]
            pair_coref = [str(pair_results[0]), str(pred_recognized_event_pairs[pair_id][0])]
            if e_i_link_len == 1:
                singleton_event_pairs[e_starts[0]].append(pair_coref[0] == pair_coref[1])
            if e_j_link_len == 1:
                singleton_event_pairs[e_starts[1]].append(pair_coref[0] == pair_coref[1])
            if mode == 'g':
                if e_i_link_len > main_link_length or e_j_link_len > main_link_length:
                    main_link_event_pairs.append(pair_coref)
            elif mode == 'ge':
                if e_i_link_len >= main_link_length or e_j_link_len >= main_link_length:
                    main_link_event_pairs.append(pair_coref)
            elif mode == 'e':
                if e_i_link_len == main_link_length or e_j_link_len == main_link_length:
                    main_link_event_pairs.append(pair_coref)
            elif mode == 'le':
                if (e_i_link_len <= main_link_length and e_i_link_len > 1) or (e_j_link_len <= main_link_length and e_j_link_len > 1):
                    main_link_event_pairs.append(pair_coref)
            elif mode == 'l':
                if (e_i_link_len < main_link_length and e_i_link_len > 1) or (e_j_link_len < main_link_length and e_j_link_len > 1):
                    main_link_event_pairs.append(pair_coref)
        for pair_id, pair_results in gold_unrecognized_event_pairs.items():
            e_starts = pair_id.split('-')
            e_i_link_len, e_j_link_len = pair_results[2], pair_results[3]
            pair_coref = [str(pair_results[0]), '2']
            if e_i_link_len == 1 and e_starts[0] not in singleton_event_pairs:
                singleton_event_pairs[e_starts[0]].append(False)
            if e_j_link_len == 1 and e_starts[1] not in singleton_event_pairs:
                singleton_event_pairs[e_starts[1]].append(False)
            if mode == 'g':
                if e_i_link_len > main_link_length or e_j_link_len > main_link_length:
                    main_link_event_pairs.append(pair_coref)
            elif mode == 'ge':
                if e_i_link_len >= main_link_length or e_j_link_len >= main_link_length:
                    main_link_event_pairs.append(pair_coref)
            elif mode == 'e':
                if e_i_link_len == main_link_length or e_j_link_len == main_link_length:
                    main_link_event_pairs.append(pair_coref)
            elif mode == 'le':
                if (e_i_link_len <= main_link_length and e_i_link_len > 1) or (e_j_link_len <= main_link_length and e_j_link_len > 1):
                    main_link_event_pairs.append(pair_coref)
            elif mode == 'l':
                if (e_i_link_len < main_link_length and e_i_link_len > 1) or (e_j_link_len < main_link_length and e_j_link_len > 1):
                    main_link_event_pairs.append(pair_coref)
        
        for pair_id, pair_results in pred_wrong_event_pairs.items():
            e_starts = pair_id.split('-')
            if e_starts[0] in singleton_event_pairs:
                singleton_event_pairs[e_starts[0]].append(pair_results[0] == 0)
            if e_starts[1] in singleton_event_pairs: 
                singleton_event_pairs[e_starts[1]].append(pair_results[0] == 0)
    
    mode_str = {'g': '>', 'ge': '>=', 'e': '==', 'le': '<=', 'l': '<'}[mode]
    metrics = {}
    y_true, y_pred = [res[0] for res in main_link_event_pairs], [res[1] for res in main_link_event_pairs]
    metrics[f'Main Link ({mode_str}{main_link_length})'] = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)['1']
    wrong_num = sum([False in singleton_coref_correct for singleton_coref_correct in singleton_event_pairs.values()])
    print(wrong_num)
    print(len(singleton_event_pairs))
    metrics['Singleton Acc'] = (len(singleton_event_pairs) - wrong_num) / len(singleton_event_pairs) * 100
    return metrics


# print(all_metrics(gold_coref_file, pred_coref_file))
# print(different_distance_metrics(gold_coref_file, pred_coref_file))
print(main_link_metrics(gold_coref_file, pred_coref_file, main_link_length=10, mode='ge'))
