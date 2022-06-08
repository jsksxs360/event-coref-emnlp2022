from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os

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
    def __init__(self, data_file, include_mention_context=False, ):
        self.vocab = vocab
        self.include_mention_context = include_mention_context
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
                    if self.include_mention_context:
                        event_mentions.append(event_mention)
                        word_dists.append(self._get_word_dist(event_mention))
                        mention_char_pos.append([
                            len(before) + (1 if len(before) > 0 else 0) + e['sent_start'], 
                            len(before) + (1 if len(before) > 0 else 0) + e['sent_start'] + len(e['trigger']) - 1
                        ])
                    else:
                        event_mentions.append(sentences[e['sent_idx']]['text'])
                        word_dists.append(self._get_word_dist(event_mention))
                        mention_char_pos.append([
                            e['sent_start'], e['sent_start'] + len(e['trigger']) - 1
                        ])
                Data.append({
                    'id': sample['doc_id'], 
                    'document': sample['document'], 
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
    new_sent, new_char_start, new_char_end = before + ' ' + trigger + ' ' + after, len(before) + 1, len(before) + len(trigger)
    assert new_sent[new_char_start:new_char_end+1] == trigger
    return new_sent, new_char_start, new_char_end

def get_dataLoader(args, dataset, tokenizer, mention_tokenizer=None, batch_size=None, shuffle=False, collote_fn_type='chunk'):

    def collote_fn_without_global(batch_samples):
        batch_events = []
        batch_mentions, batch_mention_pos = [], []
        batch_event_dists = []
        for sample in batch_samples:
            batch_events.append(sample['events'])
            batch_mentions.append(sample['event_mentions'])
            batch_mention_pos.append(sample['mention_char_pos'])
            batch_event_dists.append(sample['word_dists'])
        
        batch_mention_inputs_with_mask = []
        batch_mention_events = []
        batch_event_cluster_id = []
        batch_event_subtypes = []
        for events, mentions, mention_poss in zip(batch_events, batch_mentions, batch_mention_pos):
            new_mentions = []
            mention_events = []
            event_cluster_id = []
            event_subtypes = []
            for event, mention, mention_pos in zip(events, mentions, mention_poss):
                mention_char_start, mention_char_end = mention_pos
                mention, mention_char_start, mention_char_end = cut_sent(
                    mention, mention_char_start, mention_char_end, args.max_mention_length
                )
                mention_encoding = mention_tokenizer(mention, max_length=args.max_mention_length, truncation=True)
                mention_token_start = mention_encoding.char_to_token(mention_char_start)
                if not mention_token_start:
                    mention_token_start = mention_encoding.char_to_token(mention_char_start + 1)
                mention_token_end = mention_encoding.char_to_token(mention_char_end)
                assert mention_token_start and mention_token_end
                new_mentions.append(mention)
                mention_events.append([mention_token_start, mention_token_end])
                event_cluster_id.append(event['cluster_id'])
                event_subtypes.append(event['subtype'])
            batch_mention_inputs_with_mask.append(
                mention_tokenizer(
                    new_mentions, 
                    max_length=args.max_mention_length, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
            )
            batch_mention_events.append(mention_events)
            batch_event_cluster_id.append(event_cluster_id)
            batch_event_subtypes.append(event_subtypes)
        for b_idx in range(len(batch_event_subtypes)):
            for e_idx, (e_start, e_end) in enumerate(batch_mention_events[b_idx]):
                batch_mention_inputs_with_mask[b_idx]['input_ids'][e_idx][e_start:e_end+1] = mention_tokenizer.mask_token_id
        return {
            'batch_mention_inputs_with_mask': batch_mention_inputs_with_mask, 
            'batch_mention_events': batch_mention_events, 
            'batch_event_dists': batch_event_dists, 
            'batch_event_cluster_ids': batch_event_cluster_id, 
            'batch_event_subtypes': batch_event_subtypes
        }
    
    def collote_fn_chunk(batch_samples):
        batch_events = []
        batch_document = []
        batch_mentions, batch_mention_pos = [], []
        batch_event_dists = []
        for sample in batch_samples:
            batch_events.append(sample['events'])
            batch_document.append(sample['document'])
            batch_mentions.append(sample['event_mentions'])
            batch_mention_pos.append(sample['mention_char_pos'])
            batch_event_dists.append(sample['word_dists'])
        
        batch_mention_inputs = []
        batch_mention_events = []
        batch_mention_inputs_with_mask = []
        batch_mention_events_with_mask = []
        batch_event_cluster_id = []
        batch_event_subtypes = []
        for events, document, mentions, mention_poss in zip(batch_events, batch_document, batch_mentions, batch_mention_pos):
            new_mentions = []
            mention_events = []
            new_mentions_with_mask = []
            mention_events_with_mask = []
            event_cluster_id = []
            event_subtypes = []
            for event, mention, mention_pos in zip(events, mentions, mention_poss):
                mention_char_start, mention_char_end = mention_pos
                global_mention, global_mention_char_start, global_mention_char_end = cut_sent(
                    document, event['char_start'], event['char_end'], args.max_seq_length
                )
                encoding = tokenizer(global_mention, max_length=args.max_seq_length, truncation=True)
                token_start = encoding.char_to_token(global_mention_char_start)
                if not token_start:
                    token_start = encoding.char_to_token(global_mention_char_start + 1)
                token_end = encoding.char_to_token(global_mention_char_end)
                assert token_start and token_end
                local_mention, local_mention_char_start, local_mention_char_end = cut_sent(
                    mention, mention_char_start, mention_char_end, args.max_mention_length
                )
                mention_encoding = mention_tokenizer(local_mention, max_length=args.max_mention_length, truncation=True)
                mention_token_start = mention_encoding.char_to_token(local_mention_char_start)
                if not mention_token_start:
                    mention_token_start = mention_encoding.char_to_token(local_mention_char_start + 1)
                mention_token_end = mention_encoding.char_to_token(local_mention_char_end)
                assert mention_token_start and mention_token_end
                new_mentions.append(global_mention)
                mention_events.append([token_start, token_end])
                new_mentions_with_mask.append(local_mention)
                mention_events_with_mask.append([mention_token_start, mention_token_end])
                event_cluster_id.append(event['cluster_id'])
                event_subtypes.append(event['subtype'])
            batch_mention_inputs.append(
                tokenizer(
                    new_mentions, 
                    max_length=args.max_seq_length, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
            )
            batch_mention_events.append(mention_events)
            batch_mention_inputs_with_mask.append(
                mention_tokenizer(
                    new_mentions_with_mask, 
                    max_length=args.max_mention_length, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
            )
            batch_mention_events_with_mask.append(mention_events_with_mask)
            batch_event_cluster_id.append(event_cluster_id)
            batch_event_subtypes.append(event_subtypes)
        for b_idx in range(len(batch_event_subtypes)):
            for e_idx, (e_start, e_end) in enumerate(batch_mention_events_with_mask[b_idx]):
                batch_mention_inputs_with_mask[b_idx]['input_ids'][e_idx][e_start:e_end+1] = mention_tokenizer.mask_token_id
        return {
            'batch_mention_inputs': batch_mention_inputs, 
            'batch_mention_events': batch_mention_events, 
            'batch_mention_inputs_with_mask': batch_mention_inputs_with_mask, 
            'batch_mention_events_with_mask': batch_mention_events_with_mask, 
            'batch_event_dists': batch_event_dists, 
            'batch_event_cluster_ids': batch_event_cluster_id, 
            'batch_event_subtypes': batch_event_subtypes
        }

    if collote_fn_type == 'without_global':
        select_collote_fn = collote_fn_without_global
    elif collote_fn_type == 'chunk':
        select_collote_fn = collote_fn_chunk

    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle, 
        collate_fn=select_collote_fn
    )
