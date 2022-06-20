import json
import os
from collections import namedtuple, defaultdict

Sentence = namedtuple("Sentence", ["start", "text"])
kbp_sent_dic = defaultdict(list) # {filename: [Sentence]}
with open(os.path.join('../../data/kbp_sent.txt'), 'rt', encoding='utf-8') as sents:
    for line in sents:
        doc_id, start, text = line.strip().split('\t')
        kbp_sent_dic[doc_id].append(Sentence(int(start), text))

def get_event_sent_idx(e_start, e_end, sents):
    for sent_idx, sent in enumerate(sents):
        sent_end = sent.start + len(sent.text) - 1
        if e_start >= sent.start and e_end <= sent_end:
            return sent_idx
    return None

def get_gold_corefs(gold_test_file):

    def _get_event_cluster_id_and_link_len(event_id, clusters):
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id'], len(cluster['events'])
        return None, None

    gold_dict = {}
    with open(gold_test_file, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            clusters = sample['clusters']
            events = sample['events']
            event_pairs = {} # {e_i_start-e_j_start: (coref, sent_dist, e_i_link_len, e_j_link_len)}
            for i in range(len(events) - 1):
                e_i_start = events[i]['start']
                e_i_cluster_id, e_i_link_len = _get_event_cluster_id_and_link_len(events[i]['event_id'], clusters)
                assert e_i_cluster_id is not None
                e_i_sent_idx = events[i]['sent_idx']
                for j in range(i + 1, len(events)):
                    e_j_start = events[j]['start']
                    e_j_cluster_id, e_j_link_len = _get_event_cluster_id_and_link_len(events[j]['event_id'], clusters)
                    assert e_j_cluster_id is not None
                    e_j_sent_idx = events[j]['sent_idx']
                    event_pairs[f'{e_i_start}-{e_j_start}'] = [
                        1 if e_i_cluster_id == e_j_cluster_id else 0, abs(int(e_i_sent_idx) - int(e_j_sent_idx)), e_i_link_len, e_j_link_len
                    ]
            gold_dict[sample['doc_id']] = event_pairs
    return gold_dict

def get_pred_coref_results(pred_file_path):
    pred_dict = {}
    with open(pred_file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            sents = kbp_sent_dic[sample['doc_id']]
            events = sample['events']
            pred_labels = sample['pred_label']
            event_pairs = {} # {e_i_start-e_j_start: (coref, sent_dist, e_i_link_len, e_j_link_len)}
            event_pair_idx = -1
            for i in range(len(events) - 1):
                e_i_start = events[i]['start']
                e_i_sent_idx = get_event_sent_idx(events[i]['start'], events[i]['end'], sents)
                assert e_i_sent_idx is not None
                for j in range(i + 1, len(events)):
                    event_pair_idx += 1
                    e_j_start = events[j]['start']
                    e_j_sent_idx = get_event_sent_idx(events[j]['start'], events[j]['end'], sents)
                    assert e_j_sent_idx is not None
                    event_pairs[f'{e_i_start}-{e_j_start}'] = [pred_labels[event_pair_idx], abs(int(e_i_sent_idx) - int(e_j_sent_idx)), 0, 0]
            pred_dict[sample['doc_id']] = event_pairs
    return pred_dict

def get_event_pair_set(gold_coref_file, pred_coref_file):

    gold_coref_results = get_gold_corefs(gold_coref_file)
    pred_coref_results = get_pred_coref_results(pred_coref_file)

    new_gold_coref_results = {}
    for doc_id, event_pairs in gold_coref_results.items():
        pred_event_pairs = pred_coref_results[doc_id]
        unrecognized_event_pairs = {}
        recognized_event_pairs = {}
        for pair_id, results in event_pairs.items():
            if pair_id in pred_event_pairs:
                recognized_event_pairs[pair_id] = results
            else:
                unrecognized_event_pairs[pair_id] = results
        new_gold_coref_results[doc_id] = {
            'unrecognized_event_pairs': unrecognized_event_pairs, 
            'recognized_event_pairs': recognized_event_pairs
        }
    new_pred_coref_results = {}
    for doc_id, event_pairs in pred_coref_results.items():
        gold_event_pairs = gold_coref_results[doc_id]
        recognized_event_pairs = {}
        wrong_event_pairs = {}
        for pair_id, results in event_pairs.items():
            if pair_id in gold_event_pairs:
                recognized_event_pairs[pair_id] = results
            else:
                wrong_event_pairs[pair_id] = results
        new_pred_coref_results[doc_id] = {
            'recognized_event_pairs': recognized_event_pairs, 
            'wrong_event_pairs': wrong_event_pairs
        }

    return new_gold_coref_results, new_pred_coref_results
