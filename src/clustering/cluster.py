from collections import OrderedDict

def rescoring(events, pred_labels:list, reward=0.8, penalty=0.8):
    event_pairs = [
        str(events[i]['start']) + '-' + str(events[j]['start'])
        for i in range(len(events) - 1) for j in range(i + 1, len(events))
    ]
    coref_event_pairs = [event_pair for event_pair, pred in zip(event_pairs, pred_labels) if pred == 1]
    coref = OrderedDict([(event_pair, 1 if pred == 1 else -1) for event_pair, pred in zip(event_pairs, pred_labels)])
    for i in range(len(events) - 1):
        for j in range(i + 1, len(events)):
            for k in range(len(events)):
                if k == i or k == j:
                    continue
                event_i, event_j, event_k = events[i]['start'], events[j]['start'], events[k]['start']
                coref_i_k = (f'{event_k}-{event_i}' if k < i else f'{event_i}-{event_k}') in coref_event_pairs
                coref_j_k = (f'{event_k}-{event_j}' if k < j else f'{event_j}-{event_k}') in coref_event_pairs
                if coref_i_k and coref_j_k:
                    coref[f'{event_i}-{event_j}'] += reward
                elif coref_i_k != coref_j_k:
                    coref[f'{event_i}-{event_j}'] -= penalty
    return events, [1 if score > 0 else 0 for score in coref.values()]

def clustering_greedy(events, pred_labels:list, rescore=True, rescore_reward=0.8, rescore_penalty=0.8):
    '''
    As long as there is a pair of events coreference 
    between any two event chains, merge them.
    '''
    def need_merge(set_1, set_2, coref_event_pair_set):
        for e1 in set_1:
            for e2 in set_2:
                if f'{e1}-{e2}' in coref_event_pair_set:
                    return True
        return False

    def find_merge_position(cluster_list, coref_event_pairs):
        for i in range(len(cluster_list) - 1):
            for j in range(i + 1, len(cluster_list)):
                if need_merge(cluster_list[i], cluster_list[j], coref_event_pairs):
                    return i, j
        return -1, -1

    assert len(pred_labels) == len(events) * (len(events) - 1) / 2
    if rescore:
        events, pred_labels = rescoring(events, pred_labels, rescore_reward, rescore_penalty)
    event_pairs = [
        str(events[i]['start']) + '-' + str(events[j]['start'])
        for i in range(len(events) - 1) for j in range(i + 1, len(events))
    ]
    coref_event_pairs = [event_pair for event_pair, pred in zip(event_pairs, pred_labels) if pred == 1]
    cluster_list = []
    for event in events: # init each link as an event
        cluster_list.append(set([event['start']]))
    while True:
        i, j = find_merge_position(cluster_list, coref_event_pairs)
        if i == -1: # no cluster can be merged
            break
        cluster_list[i] |= cluster_list[j]
        del cluster_list[j]
    return cluster_list
