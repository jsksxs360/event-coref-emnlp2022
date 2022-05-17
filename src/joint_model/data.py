from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import torch

SUBTYPES = [ # 18 subtypes
    'artifact', 'transferownership', 'transaction', 'broadcast', 'contact', 'demonstrate', \
    'injure', 'transfermoney', 'transportartifact', 'attack', 'meet', 'elect', \
    'endposition', 'correspondence', 'arrestjail', 'startposition', 'transportperson', 'die'
]
VOCAB_SIZE = 500

id2subtype = {idx: c for idx, c in enumerate(SUBTYPES, start=1)}
subtype2id = {v: k for k, v in id2subtype.items()}
total_word_count = defaultdict(int)
if os.path.exists('../../data/kbp_vocab.txt'):
    print(f'loading vocab file...')
    with open('../../data/kbp_vocab.txt', 'rt') as f:
        vocab = json.loads(f.readlines()[0].strip())
else:
    print(f'creating vocab file...')
    with open('../../data/kbp_word_count.txt', 'rt') as f:
        for line in f:
            word_count = json.loads(line.strip().split('\t')[1])
            for word, num in word_count.items():
                total_word_count[word] += num
    vocab = [
        w for w, _ in 
        sorted(total_word_count.items(), key=lambda t: t[1], reverse=True)[:VOCAB_SIZE]
    ]
    with open('../../data/kbp_vocab.txt', 'wt') as f:
        f.write(json.dumps(vocab))

class KBPCoref(Dataset):
    def __init__(self, data_file):
        self.vocab = vocab
        self.data = self.load_data(data_file)

    def _get_word_dist(self, event_mention:str):
        event_mention = event_mention.lower()
        return [1 if w in event_mention else 0 for w in self.vocab]
    
    def _get_event_cluster_id(self, event_id:str, clusters:list) -> str:
        for cluster in clusters:
            if event_id in cluster['events']:
                return cluster['hopper_id']
        return None

    def load_data(self, data_file):
        Data = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                clusters = sample['clusters']
                sentences = sample['sentences']
                td_tags = [
                    {
                        'char_start': e['start'], 
                        'char_end': e['start'] + len(e['trigger']) - 1, 
                        'trigger': e['trigger'], 
                        'subtype': e['subtype']
                    }
                    for e in sample['events'] if e['subtype'] in SUBTYPES
                ]
                events = [
                    {
                        'event_id': e['event_id'], 
                        'char_start': e['start'], 
                        'char_end': e['start'] + len(e['trigger']) - 1, 
                        'trigger': e['trigger'], 
                        'subtype': subtype2id.get(e['subtype'], 0), # 0 - 'other'
                        'cluster_id': self._get_event_cluster_id(e['event_id'], clusters)
                    }
                    for e in sample['events']
                ]
                event_mentions, mention_char_pos, word_dists = [], [], []
                for e in sample['events']:
                    before = sentences[e['sent_idx'] - 1]['text'] if e['sent_idx'] > 0 else ''
                    after = sentences[e['sent_idx'] + 1]['text'] if e['sent_idx'] < len(sentences) - 1 else ''
                    event_mention = before + (' ' if len(before) > 0 else '') + sentences[e['sent_idx']]['text'] + ' ' + after
                    event_mentions.append(sentences[e['sent_idx']]['text'])
                    word_dists.append(self._get_word_dist(event_mention))
                    mention_char_pos.append([
                        e['sent_start'], e['sent_start'] + len(e['trigger']) - 1
                    ])
                Data.append({
                    'id': sample['doc_id'], 
                    'document': sample['document'], 
                    'td_tags': td_tags, 
                    'events': events, 
                    'event_mentions': event_mentions, 
                    'mention_char_pos': mention_char_pos, 
                    'word_dists': word_dists
                })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def cut_sent(sent, e_char_start, e_char_end, max_length):
    span_max_length = (max_length - 50) // 2
    before = ' '.join([c for c in sent[:e_char_start].split(' ') if c != ''][-span_max_length:]).strip()
    trigger = sent[e_char_start:e_char_end+1]
    after = ' '.join([c for c in sent[e_char_end+1:].split(' ') if c != ''][:span_max_length]).strip()
    return before + ' ' + trigger + ' ' + after, len(before) + 1, len(before) + len(trigger)

def get_dataLoader(args, dataset, tokenizer, mention_tokenizer=None, batch_size=None, shuffle=False, collote_fn_type='normal'):

    def collote_fn(batch_samples):
        batch_sentences, batch_td_tags, batch_events  = [], [], []
        for sample in batch_samples:
            batch_sentences.append(sample['document'])
            batch_td_tags.append(sample['td_tags'])
            batch_events.append(sample['events'])
        batch_inputs = tokenizer(
            batch_sentences, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        # trigger detection
        batch_td_label = np.zeros(batch_inputs['input_ids'].shape, dtype=int)
        for b_idx, sentence in enumerate(batch_sentences):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            for e in batch_td_tags[b_idx]:
                token_start = encoding.char_to_token(e['char_start'])
                if not token_start:
                    token_start = encoding.char_to_token(e['char_start'] + 1)
                token_end = encoding.char_to_token(e['char_end'])
                if not token_start or not token_end:
                    continue
                tag = e['subtype']
                batch_td_label[b_idx][token_start] = args.label2id[f"B-{tag}"]
                batch_td_label[b_idx][token_start + 1:token_end + 1] = args.label2id[f"I-{tag}"]
        # event coref
        batch_filtered_events = []
        batch_filtered_event_cluster_id = []
        for sentence, events in zip(batch_sentences, batch_events):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            filtered_events = []
            filtered_event_cluster_id = []
            for e in events:
                token_start = encoding.char_to_token(e['char_start'])
                if not token_start:
                    token_start = encoding.char_to_token(e['char_start'] + 1)
                token_end = encoding.char_to_token(e['char_end'])
                if not token_start or not token_end:
                    continue
                filtered_events.append([token_start, token_end])
                filtered_event_cluster_id.append(e['cluster_id'])
            batch_filtered_events.append(filtered_events)
            batch_filtered_event_cluster_id.append(filtered_event_cluster_id)
        return {
            'batch_inputs': batch_inputs, 
            'batch_td_labels': torch.tensor(batch_td_label), 
            'batch_events': batch_filtered_events, 
            'batch_event_cluster_ids': batch_filtered_event_cluster_id
        }
    
    if collote_fn_type == 'normal':
        select_collote_fn = collote_fn
    
    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle, 
        collate_fn=select_collote_fn
    )