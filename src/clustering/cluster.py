from collections import OrderedDict, defaultdict

def clustering_greedy(events, pred_labels:list):
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

    if len(events) > 1:
        assert len(pred_labels) == len(events) * (len(events) - 1) / 2
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

def clustering_rescore(events, pred_labels:list, reward=0.8, penalty=0.8):
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
    coref = OrderedDict([(event_pair, score) for event_pair, score in coref.items() if score > 0])
    sorted_coref = sorted(coref.items(), key=lambda x:x[1], reverse=True)
    cluster_id = 0
    events_cluster_ids = {str(event['start']):-1 for event in events} # {event:cluster_id}
    for event_pair, _ in sorted_coref:
        e_i, e_j = event_pair.split('-')
        if events_cluster_ids[e_i] == events_cluster_ids[e_j] == -1:
            events_cluster_ids[e_i] = events_cluster_ids[e_j] = cluster_id
            cluster_id += 1
        elif events_cluster_ids[e_i] == -1:
            events_cluster_ids[e_i] = events_cluster_ids[e_j]
        elif events_cluster_ids[e_j] == -1:
            events_cluster_ids[e_j] = events_cluster_ids[e_j]
    for event, c_id in events_cluster_ids.items():
        if c_id == -1:
            events_cluster_ids[event] = cluster_id
            cluster_id += 1
    cluster_list = defaultdict(set)
    for event, c_id in events_cluster_ids.items():
        cluster_list[c_id].add(event)
    return [v for v in cluster_list.values()]

def clustering(events, pred_labels:list, mode='rescore', rescore_reward=0.8, rescore_penalty=0.8):
    assert mode in ['greedy', 'rescore']
    if mode == 'rescore':
        return clustering_rescore(events, pred_labels, rescore_reward, rescore_penalty)
    elif mode == 'greedy':
        return clustering_greedy(events, pred_labels)
