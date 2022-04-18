import collections
import json

def get_pred_coref_results(pred_file_path, ):
    pred_results = {} # {doc_id: {'events': event_list, 'pred_labels': pred_coref_labels}}
    with open(pred_file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            pred_results[sample['doc_id']] = {
                'events': sample['events'], 
                'pred_labels': sample['pred_label']
            }
    return pred_results

def create_golden_conll_file(test_file_path, conll_file_path):

    def get_event_cluster_idx(event_id:str, clusters):
        for idx, cluster in enumerate(clusters):
            if event_id in cluster['events']:
                return idx
        print('ERROR!')
        return None

    with open(test_file_path, 'rt', encoding='utf-8') as f_in, \
         open(conll_file_path, 'wt', encoding='utf-8') as f_out:
        for line in f_in:
            sample = json.loads(line.strip())
            doc_id = sample['doc_id']
            f_out.write(f'#begin document ({doc_id});\n')
            clusters = sample['clusters']
            for event in sample['events']:
                cluster_idx = get_event_cluster_idx(event['event_id'], clusters)
                start = event['start']
                f_out.write(f'{doc_id}\t{start}\txxx\t({cluster_idx})\n')
            f_out.write('#end document\n')

def create_pred_conll_file(cluster_dict:dict, golden_conll_filepath:str, conll_filepath:str, no_repeat=True):
    '''
    # Args:
        - cluster_dict: {doc_id: [cluster_set_1, cluster_set_2, ...]}
    '''
    new_cluster_dict = {} # {doc_id: {event: cluster_idx}}
    for doc_id, cluster_list in cluster_dict.items():
        event_cluster_idx = {} # {event: cluster_idx}
        for c_idx, cluster in enumerate(cluster_list):
            for event in cluster:
                event_cluster_idx[str(event)] = c_idx
        new_cluster_dict[doc_id] = event_cluster_idx
    golden_file_dic = collections.OrderedDict() # {doc_id: [event_1, event_2, ...]}
    with open(golden_conll_filepath, 'rt', encoding='utf-8') as f_in:
        for line in f_in:
            if line.startswith('#begin'):
                doc_id = line.replace('#begin document (', '').replace(');', '').strip()
                golden_file_dic[doc_id] = []
            elif line.startswith('#end document'):
                continue
            else:
                _, event, _, _ = line.strip().split('\t')
                golden_file_dic[doc_id].append(event)
    with open(conll_filepath, 'wt', encoding='utf-8') as f_out:
        for doc_id, event_list in golden_file_dic.items():
            event_cluster_idx = new_cluster_dict[doc_id]
            f_out.write('#begin document (' + doc_id + ');\n')
            if no_repeat:
                finish_events = set()
                for event in event_list:
                    if event in event_cluster_idx and event not in finish_events:
                        cluster_idx = event_cluster_idx[event]
                        f_out.write(f'{doc_id}\t{event}\txxx\t({cluster_idx})\n')
                    else:
                        f_out.write(f'{doc_id}\tnull\tnull\tnull\n')
                    finish_events.add(event)
            else:
                for event in event_list:
                    if event in event_cluster_idx:
                        cluster_idx = event_cluster_idx[event]
                        f_out.write(f'{doc_id}\t{event}\txxx\t({cluster_idx})\n')
                    else:
                        f_out.write(f'{doc_id}\tnull\tnull\tnull\n')
            for event, cluster_idx in event_cluster_idx.items():
                if event in event_list:
                    continue
                f_out.write(f'{doc_id}\t{event}\txxx\t({cluster_idx})\n')
            f_out.write('#end document\n')
