import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger("Utils")
filter_logger = logging.getLogger("Filter")

def print_data_statistic(doc_list, dataset=''):
    doc_num = len(doc_list)
    event_num = sum([len(doc['events']) for doc in doc_list])
    cluster_num = sum([len(doc['clusters']) for doc in doc_list])
    singleton_num = sum([1 if len(cluster['events']) == 1  else 0 
                                for doc in doc_list for cluster in doc['clusters']])
    logger.info(f"KBP {dataset} - Doc: {doc_num} | Event: {event_num} | Cluster: {cluster_num} | Singleton: {singleton_num}")

def check_event_conflict(doc_list):
    for doc in doc_list:
        event_list = doc['events']
        event_list.sort(key=lambda x:x['start'])
        if len(event_list) < 2:
            continue
        for idx in range(len(event_list)-1):
            if (
                (
                    event_list[idx]['start'] == event_list[idx+1]['start'] and
                    event_list[idx]['trigger'] == event_list[idx+1]['trigger']
                ) or 
                (
                    event_list[idx]['start'] + len(event_list[idx]['trigger']) >
                    event_list[idx+1]['start']
                )
            ):
                logger.error('{}: ({})[{}] VS ({})[{}]'.format(doc['doc_id'], 
                    event_list[idx]['start'], event_list[idx]['trigger'], 
                    event_list[idx+1]['start'], event_list[idx+1]['trigger']))

def filter_events(doc_list, dataset=''):
    same = 0
    overlapping = 0
    cluster_num_filtered = 0
    for doc in doc_list:
        event_list = doc['events']
        event_list.sort(key=lambda x:x['start'])
        event_filtered = []
        if len(event_list) < 2:
            continue
        new_event_list, should_add = [], True
        for idx in range(len(event_list)-1):
            if (event_list[idx]['start'] == event_list[idx+1]['start'] and
                event_list[idx]['trigger'] == event_list[idx+1]['trigger']
            ):
                event_filtered.append(event_list[idx]['event_id'])
                same += 1
                continue
            if (event_list[idx]['start'] + len(event_list[idx]['trigger']) >
                event_list[idx+1]['start']
            ):
                overlapping += 1
                if len(event_list[idx]['trigger']) < len(event_list[idx+1]['trigger']):
                    new_event_list.append(event_list[idx])
                    should_add = False
                else:
                    event_filtered.append(event_list[idx]['event_id'])
                continue
            if should_add:
                new_event_list.append(event_list[idx])
            else:
                event_filtered.append(event_list[idx]['event_id'])
                should_add = True
        if should_add:
            new_event_list.append(event_list[-1])
        doc['events'] =  new_event_list
        new_clusters = []
        for cluster in doc['clusters']:
            new_events = [event_id for event_id in cluster['events'] if event_id not in event_filtered]
            if len(new_events) == 0:
                cluster_num_filtered += 1
                continue
            new_clusters.append({
                'hopper_id': cluster['hopper_id'], 
                'events': new_events
            })
        doc['clusters'] = new_clusters
    filter_logger.info(f'KBP {dataset} event filtered: {same + overlapping} (same {same} / overlapping {overlapping})')
    filter_logger.info(f'KBP {dataset} cluster filtered: {cluster_num_filtered}')
    return doc_list