import collections
from collections import namedtuple
import xml.etree.ElementTree as ET
import os
import re
from typing import Dict, List, Tuple
import logging
import json
import numpy as np
from itertools import combinations
import sys
sys.path.append('../')
from data.utils import print_data_statistic, filter_events, check_event_conflict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger("Convert")

SENT_FILE = 'kbp_sent.txt'
DATA_DIRS = {
    '2015': [
        'LDC_TAC_KBP/LDC2015E29/data/ere/mpdfxml', 
        'LDC_TAC_KBP/LDC2015E68/data/ere', 
        'LDC_TAC_KBP/LDC2017E02/data/2015/training/event_hopper', 
        'LDC_TAC_KBP/LDC2017E02/data/2015/eval/hopper'
    ],
    '2016': [
        'LDC_TAC_KBP/LDC2017E02/data/2016/eval/eng/nw/ere', 
        'LDC_TAC_KBP/LDC2017E02/data/2016/eval/eng/df/ere'
    ],
    '2017': [
        'LDC_TAC_KBP/LDC2017E54/data/eng/nw/ere', 
        'LDC_TAC_KBP/LDC2017E54/data/eng/df/ere'
    ]
}

Sentence = namedtuple("Sentence", ["start", "text"])
Filename = namedtuple("Filename", ["doc_id", "file_path"])

def get_KBP_sents(sent_file_path:str) -> Dict[str, List[Sentence]]:
    '''get sentences in the KBP dataset
    # Returns: 
        - sentence dictionary: {filename: [Sentence]}
    '''
    sent_dic = collections.defaultdict(list)
    with open(sent_file_path, 'rt', encoding='utf-8') as sents:
        for line in sents:
            doc_id, start, text = line.strip().split('\t')
            sent_dic[doc_id].append(Sentence(int(start), text))
    for sents in sent_dic.values():
        sents.sort(key=lambda x:x.start)
    return sent_dic

def get_KBP_filenames(version:str) -> List[Filename]:
    '''get KBP filenames
    # Args:
        - version: 2015 / 2016 / 2017
    # Return:
        - filename list: [Filename]
    '''
    assert version in ['2015', '2016', '2017']
    filename_list = []
    for folder in DATA_DIRS[version]:
        filename_list += [
            Filename(
                re.sub('\.event_hoppers\.xml|\.rich_ere\.xml', '', filename), 
                os.path.join(folder, filename)
            ) for filename in os.listdir(folder)
        ]
    return filename_list

def create_new_document(sent_list:List[Sentence]) -> str:
    '''create new source document
    '''
    document = ''
    end = 0
    for sent in sent_list:
        assert sent.start >= end
        document += ' ' * (sent.start - end)
        document += sent.text
        end = sent.start + len(sent.text)
    for sent in sent_list: # check
        assert document[sent.start:sent.start+len(sent.text)] == sent.text
    return document

def find_event_sent(doc_id, event_start, trigger, sent_list) -> Tuple[int, int]:
    '''find out which sentence the event come from
    '''
    for idx, sent in enumerate(sent_list):
        s_start, s_end = sent.start, sent.start + len(sent.text) - 1
        if s_start <= event_start <= s_end:
            e_s_start = event_start - s_start
            assert sent.text[e_s_start:e_s_start+len(trigger)] == trigger
            return idx, event_start - s_start
    print(doc_id)
    print(event_start, trigger, '\n')
    for sent in sent_list:
        print(sent.start, sent.start + len(sent.text) - 1)
    return None

def update_trigger(text, trigger, offset):
    punc_set = set('#$%&+=@.,;!?*\\~\'\n\r\t()[]|/’-:{<>}、"。，？“”')
    new_trigger = trigger
    if offset + len(trigger) < len(text) and text[offset + len(trigger)] != ' ' and text[offset + len(trigger)] not in punc_set:
        for c in text[offset + len(trigger):]:
            if c == ' ' or c in punc_set:
                break
            new_trigger += c
        new_trigger = new_trigger.strip('\n\r\t')
        new_trigger = new_trigger.strip(u'\x94')
    if new_trigger != trigger:
        logger.warning(f'update: [{trigger}]({len(trigger)}) - [{new_trigger}]({len(new_trigger)})')
    return new_trigger

def xml_parser(file_path:str, sent_list:List[Sentence]) -> Dict:
    '''KBP datafile XML parser
    # Args:
        - file_path: xml file path
        - sent_list: Sentences of file
    '''
    tree = ET.ElementTree(file=file_path)
    doc_id = re.sub('\.event_hoppers\.xml|\.rich_ere\.xml', '', os.path.split(file_path)[1])
    document = create_new_document(sent_list)
    sentence_list = [{'start': sent.start, 'text': sent.text} for sent in sent_list]
    event_list = []
    cluster_list = []
    for hopper in tree.iter(tag='hopper'):
        h_id = hopper.attrib['id']  # hopper id
        h_events = []
        for event in hopper.iter(tag='event_mention'):
            att = event.attrib
            e_id = att['id']
            e_type, e_subtype, e_realis = att['type'], att['subtype'], att['realis']
            e_trigger = event.find('trigger').text.strip()
            e_start = int(event.find('trigger').attrib['offset'])
            e_s_index, e_s_start = find_event_sent(doc_id, e_start, e_trigger, sent_list)
            e_trigger = update_trigger(sent_list[e_s_index].text, e_trigger, e_s_start)
            event_list.append({
                'event_id': e_id, 
                'start': e_start, 
                'trigger': e_trigger, 
                'type': e_type, 
                'subtype': e_subtype, 
                'realis': e_realis, 
                'sent_idx': e_s_index, 
                'sent_start': e_s_start
            })
            h_events.append(e_id)
        cluster_list.append({
            'hopper_id': h_id, 
            'events': h_events
        })
    return {
        'doc_id': doc_id, 
        'document': document, 
        'sentences': sentence_list, 
        'events': event_list, 
        'clusters': cluster_list
    }

def split_dev(doc_list:list, valid_doc_num:int, valid_event_num:int, valid_chain_num:int):
    '''split dev set from full train set
    '''
    docs_id = [doc['doc_id'] for doc in doc_list]
    docs_event_num = np.asarray([len(doc['events']) for doc in doc_list])
    docs_event_num[docs_id.index('bolt-eng-DF-170-181109-47916')] += 2
    docs_event_num[docs_id.index('bolt-eng-DF-170-181109-48534')] += 1
    docs_cluster_num = np.asarray([len(doc['clusters']) for doc in doc_list])
    logger.info(f'Train & Dev set: Doc: {len(docs_id)} | Event: {docs_event_num.sum()} | Cluster: {docs_cluster_num.sum()}')
    train_docs, dev_docs = [], []
    logger.info(f'finding the correct split...')
    for indexs in combinations(range(len(docs_id)), valid_doc_num):
        indexs = np.asarray(indexs)
        if (
            docs_event_num[indexs].sum() == valid_event_num and 
            docs_cluster_num[indexs].sum() == valid_chain_num
        ):
            logger.info(f'Done!')
            for idx, doc in enumerate(doc_list):
                if idx in indexs:
                    dev_docs.append(doc)
                else:
                    train_docs.append(doc)
            break
    return train_docs, dev_docs

if __name__ == "__main__":
    docs = collections.defaultdict(list)
    kbp_sent_list = get_KBP_sents(SENT_FILE)
    for dataset in ['2015', '2016', '2017']:
        logger.info(f"parsing xml files in KBP {dataset} ...")
        for filename in get_KBP_filenames(dataset):
            doc_results = xml_parser(filename.file_path, kbp_sent_list[filename.doc_id])
            docs[f'kbp_{dataset}'].append(doc_results)
        logger.info(f"Finished!")
        print_data_statistic(docs[f'kbp_{dataset}'], dataset)
    # split Dev set
    train_docs, dev_docs = split_dev(docs['kbp_2015'] + docs['kbp_2016'], 82, 2382, 1502)
    kbp_dataset = {
        'train': train_docs, 
        'dev': dev_docs, 
        'test': docs['kbp_2017']
    }
    for doc_list in kbp_dataset.values():
        doc_list.sort(key=lambda x:x['doc_id'])
    for dataset in ['train', 'dev', 'test']:
        logger.info(f"saving {dataset} set ...")
        dataset_doc_list = kbp_dataset[dataset]
        print_data_statistic(dataset_doc_list, dataset)
        with open(f'{dataset}.json', 'wt', encoding='utf-8') as f:
            for doc in dataset_doc_list:
                f.write(json.dumps(doc) + '\n')
        logger.info(f"Finished!")
    # filter events & clusters
    for dataset in ['train', 'dev', 'test']:
        dataset_doc_list = filter_events(kbp_dataset[dataset], dataset)
        check_event_conflict(dataset_doc_list)
        print_data_statistic(dataset_doc_list, dataset)
        logger.info(f"saving filtered {dataset} set ...")
        with open(f'{dataset}_filtered.json', 'wt', encoding='utf-8') as f:
            for doc in dataset_doc_list:
                f.write(json.dumps(doc) + '\n')
        logger.info(f"Finished!")