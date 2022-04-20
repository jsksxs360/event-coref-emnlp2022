WRONG_TYPE = {
    0: 'recognize_non-coref_as_coref', 
    1: 'recognize_coref_as_non-coref'
}

def get_pretty_event(sentences, sent_idx, sent_start, trigger, context=1):
    before = ' '.join([sent['text'] for sent in sentences[max(0, sent_idx-context):sent_idx]]).strip()
    after = ' '.join([sent['text'] for sent in sentences[sent_idx+1:min(len(sentences), sent_idx+context+1)]]).strip()
    event_mention = sentences[sent_idx]['text']
    sent = event_mention[:sent_start] + '#####' + trigger + '#####' + event_mention[sent_start + len(trigger):]
    return before + ' ' + sent + ' ' + after

def find_event_by_start(events, offset):
    for event in events:
        if event['start'] == offset:
            return event
    return None

def get_coref_answer(clusters, e1_id, e2_id):
    for cluster in clusters:
        events = cluster['events']
        if e1_id in events and e2_id in events:
            return 1
        elif e1_id in events or e2_id in events:
            return 0
    return 0

def get_wrong_samples(doc_id, new_events, predictions, source_events, clusters, sentences):
    wrong_1_list, wrong_2_list = [], []
    
    idx = 0
    true_labels = []
    for i in range(len(new_events) - 1):
        for j in range(i + 1, len(new_events)):
            e1_start, e2_start = new_events[i][0], new_events[j][0]
            e1 = find_event_by_start(source_events, e1_start)
            e2 = find_event_by_start(source_events, e2_start)
            pred_coref = predictions[idx]
            idx += 1
            true_coref = get_coref_answer(clusters, e1['event_id'], e2['event_id'])
            true_labels.append(true_coref)
            if pred_coref == true_coref:
                continue
            pretty_e1 = get_pretty_event(sentences, e1['sent_idx'], e1['sent_start'], e1['trigger'])
            pretty_e2 = get_pretty_event(sentences, e2['sent_idx'], e2['sent_start'], e2['trigger'])
            if pred_coref == 1:
                wrong_1_list.append({
                    'doc_id': doc_id, 
                    'e1_start': e1_start, 
                    'e2_start': e2_start, 
                    'e1_info': pretty_e1, 
                    'e2_info': pretty_e2, 
                    'wrong_type': 0
                })
            else:
                wrong_2_list.append({
                    'doc_id': doc_id, 
                    'e1_start': e1_start, 
                    'e2_start': e2_start, 
                    'e1_info': pretty_e1, 
                    'e2_info': pretty_e2, 
                    'wrong_type': 1
                })
    return wrong_1_list, wrong_2_list, true_labels