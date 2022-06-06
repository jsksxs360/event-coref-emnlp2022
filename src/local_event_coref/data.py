from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
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

class KBPCorefPair(Dataset):
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
                events = []
                for e in sample['events']:
                    before = sentences[e['sent_idx'] - 1]['text'] if e['sent_idx'] > 0 else ''
                    after = sentences[e['sent_idx'] + 1]['text'] if e['sent_idx'] < len(sentences) - 1 else ''
                    event_mention = before + (' ' if len(before) > 0 else '') + sentences[e['sent_idx']]['text'] + ' ' + after
                    events.append({
                        'char_start': e['start'], 
                        'sent_start': e['sent_start'], 
                        'sent_end': e['sent_start'] + len(e['trigger']) - 1, 
                        'sent_text': sentences[e['sent_idx']]['text'], 
                        'word_dist': self._get_word_dist(event_mention),
                        'subtype': subtype2id.get(e['subtype'], 0), # 0 - 'other'
                        'cluster_id': self._get_event_cluster_id(e['event_id'], clusters)
                    })
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1 = events[i]
                        event_2 = events[j]
                        Data.append({
                            'id': sample['doc_id'], 
                            'e1_offset': event_1['char_start'], 
                            'e1_sen': event_1['sent_text'], 
                            'e1_start': event_1['sent_start'], 
                            'e1_end': event_1['sent_end'], 
                            'e1_dist': event_1['word_dist'], 
                            'e1_subtype': event_1['subtype'], 
                            'e2_offset': event_2['char_start'], 
                            'e2_sen': event_2['sent_text'], 
                            'e2_start': event_2['sent_start'], 
                            'e2_end': event_2['sent_end'], 
                            'e2_dist': event_2['word_dist'], 
                            'e2_subtype': event_2['subtype'], 
                            'label': 1 if event_1['cluster_id'] == event_2['cluster_id'] else 0
                        })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False, collote_fn_type='normal'):

    def _cut_sent(sent, e_char_start, e_char_end, max_length):
        before = ' '.join([c for c in sent[:e_char_start].split(' ') if c != ''][-max_length:]).strip()
        trigger = sent[e_char_start:e_char_end+1]
        after = ' '.join([c for c in sent[e_char_end+1:].split(' ') if c != ''][:max_length]).strip()
        new_sent, new_char_start, new_char_end = before + ' ' + trigger + ' ' + after, len(before) + 1, len(before) + len(trigger)
        assert new_sent[new_char_start:new_char_end+1] == trigger
        return new_sent, new_char_start, new_char_end

    max_mention_length = (args.max_seq_length - 50) // 4

    def collote_fn(batch_samples):
        batch_sen_1, batch_sen_2, batch_event_idx, batch_label  = [], [], [], []
        for sample in batch_samples:
            sen_1, e1_char_start, e1_char_end = _cut_sent(sample['e1_sen'], sample['e1_start'], sample['e1_end'], max_mention_length)
            sen_2, e2_char_start, e2_char_end = _cut_sent(sample['e2_sen'], sample['e2_start'], sample['e2_end'], max_mention_length)
            batch_sen_1.append(sen_1)
            batch_sen_2.append(sen_2)
            batch_event_idx.append(
                (e1_char_start, e1_char_end, e2_char_start, e2_char_end)
            )
            batch_label.append(sample['label'])
        batch_inputs = tokenizer(
            batch_sen_1, 
            batch_sen_2, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_e1_token_idx, batch_e2_token_idx = [], []
        for sen_1, sen_2, event_idx in zip(batch_sen_1, batch_sen_2, batch_event_idx):
            e1_char_start, e1_char_end, e2_char_start, e2_char_end = event_idx
            encoding = tokenizer(sen_1, sen_2, max_length=args.max_seq_length, truncation=True)
            e1_start = encoding.char_to_token(e1_char_start, sequence_index=0)
            if not e1_start:
                e1_start = encoding.char_to_token(e1_char_start + 1, sequence_index=0)
            e1_end = encoding.char_to_token(e1_char_end, sequence_index=0)
            e2_start = encoding.char_to_token(e2_char_start, sequence_index=1)
            if not e2_start:
                e2_start = encoding.char_to_token(e2_char_start + 1, sequence_index=1)
            e2_end = encoding.char_to_token(e2_char_end, sequence_index=1)
            assert e1_start and e1_end and e2_start and e2_end
            batch_e1_token_idx.append([[e1_start, e1_end]])
            batch_e2_token_idx.append([[e2_start, e2_end]])
        return {
            'batch_inputs': batch_inputs, 
            'batch_e1_idx': batch_e1_token_idx, 
            'batch_e2_idx': batch_e2_token_idx, 
            'labels': batch_label
        }
    
    def collote_fn_with_mask(batch_samples):
        batch_sen_1, batch_sen_2, batch_event_idx = [], [], []
        batch_label, batch_subtypes = [], []
        for sample in batch_samples:
            sen_1, e1_char_start, e1_char_end = _cut_sent(sample['e1_sen'], sample['e1_start'], sample['e1_end'], max_mention_length)
            sen_2, e2_char_start, e2_char_end = _cut_sent(sample['e2_sen'], sample['e2_start'], sample['e2_end'], max_mention_length)
            batch_sen_1.append(sen_1)
            batch_sen_2.append(sen_2)
            batch_event_idx.append(
                (e1_char_start, e1_char_end, e2_char_start, e2_char_end)
            )
            batch_label.append(sample['label'])
            batch_subtypes.append([sample['e1_subtype'], sample['e2_subtype']])
        batch_inputs = tokenizer(
            batch_sen_1, 
            batch_sen_2, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_inputs_with_mask = tokenizer(
            batch_sen_1, 
            batch_sen_2, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_e1_token_idx, batch_e2_token_idx = [], []
        for sen_1, sen_2, event_idx in zip(batch_sen_1, batch_sen_2, batch_event_idx):
            e1_char_start, e1_char_end, e2_char_start, e2_char_end = event_idx
            encoding = tokenizer(sen_1, sen_2, max_length=args.max_seq_length, truncation=True)
            e1_start = encoding.char_to_token(e1_char_start, sequence_index=0)
            if not e1_start:
                e1_start = encoding.char_to_token(e1_char_start + 1, sequence_index=0)
            e1_end = encoding.char_to_token(e1_char_end, sequence_index=0)
            e2_start = encoding.char_to_token(e2_char_start, sequence_index=1)
            if not e2_start:
                e2_start = encoding.char_to_token(e2_char_start + 1, sequence_index=1)
            e2_end = encoding.char_to_token(e2_char_end, sequence_index=1)
            assert e1_start and e1_end and e2_start and e2_end
            batch_e1_token_idx.append([[e1_start, e1_end]])
            batch_e2_token_idx.append([[e2_start, e2_end]])
        for b_idx in range(len(batch_label)):
            e1_start, e1_end = batch_e1_token_idx[b_idx][0]
            e2_start, e2_end = batch_e2_token_idx[b_idx][0]
            batch_inputs_with_mask['input_ids'][b_idx][e1_start:e1_end+1] = tokenizer.mask_token_id
            batch_inputs_with_mask['input_ids'][b_idx][e2_start:e2_end+1] = tokenizer.mask_token_id
        return {
            'batch_inputs': batch_inputs, 
            'batch_inputs_with_mask': batch_inputs_with_mask, 
            'batch_e1_idx': batch_e1_token_idx, 
            'batch_e2_idx': batch_e2_token_idx, 
            'labels': batch_label, 
            'subtypes': batch_subtypes
        }
    
    def collote_fn_with_dist(batch_samples):
        batch_sen_1, batch_sen_2, batch_event_idx = [], [], []
        batch_e1_dists, batch_e2_dists = [], []
        batch_label = []
        for sample in batch_samples:
            sen_1, e1_char_start, e1_char_end = _cut_sent(sample['e1_sen'], sample['e1_start'], sample['e1_end'], max_mention_length)
            sen_2, e2_char_start, e2_char_end = _cut_sent(sample['e2_sen'], sample['e2_start'], sample['e2_end'], max_mention_length)
            batch_sen_1.append(sen_1)
            batch_sen_2.append(sen_2)
            batch_event_idx.append(
                (e1_char_start, e1_char_end, e2_char_start, e2_char_end)
            )
            batch_e1_dists.append(sample['e1_dist'])
            batch_e2_dists.append(sample['e2_dist'])
            batch_label.append(sample['label'])
        batch_inputs = tokenizer(
            batch_sen_1, 
            batch_sen_2, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_e1_token_idx, batch_e2_token_idx = [], []
        for sen_1, sen_2, event_idx in zip(batch_sen_1, batch_sen_2, batch_event_idx):
            e1_char_start, e1_char_end, e2_char_start, e2_char_end = event_idx
            encoding = tokenizer(sen_1, sen_2, max_length=args.max_seq_length, truncation=True)
            e1_start = encoding.char_to_token(e1_char_start, sequence_index=0)
            if not e1_start:
                e1_start = encoding.char_to_token(e1_char_start + 1, sequence_index=0)
            e1_end = encoding.char_to_token(e1_char_end, sequence_index=0)
            e2_start = encoding.char_to_token(e2_char_start, sequence_index=1)
            if not e2_start:
                e2_start = encoding.char_to_token(e2_char_start + 1, sequence_index=1)
            e2_end = encoding.char_to_token(e2_char_end, sequence_index=1)
            assert e1_start and e1_end and e2_start and e2_end
            batch_e1_token_idx.append([[e1_start, e1_end]])
            batch_e2_token_idx.append([[e2_start, e2_end]])
        return {
            'batch_inputs': batch_inputs, 
            'batch_e1_idx': batch_e1_token_idx, 
            'batch_e2_idx': batch_e2_token_idx, 
            'batch_e1_dists': batch_e1_dists, 
            'batch_e2_dists': batch_e2_dists, 
            'labels': batch_label
        }
    
    def collote_fn_with_mask_dist(batch_samples):
        batch_sen_1, batch_sen_2, batch_event_idx = [], [], []
        batch_e1_dists, batch_e2_dists = [], []
        batch_subtypes, batch_label = []
        for sample in batch_samples:
            sen_1, e1_char_start, e1_char_end = _cut_sent(sample['e1_sen'], sample['e1_start'], sample['e1_end'], max_mention_length)
            sen_2, e2_char_start, e2_char_end = _cut_sent(sample['e2_sen'], sample['e2_start'], sample['e2_end'], max_mention_length)
            batch_sen_1.append(sen_1)
            batch_sen_2.append(sen_2)
            batch_event_idx.append(
                (e1_char_start, e1_char_end, e2_char_start, e2_char_end)
            )
            batch_e1_dists.append(sample['e1_dist'])
            batch_e2_dists.append(sample['e2_dist'])
            batch_subtypes.append([sample['e1_subtype'], sample['e2_subtype']])
            batch_label.append(sample['label'])
        batch_inputs = tokenizer(
            batch_sen_1, 
            batch_sen_2, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_inputs_with_mask = tokenizer(
            batch_sen_1, 
            batch_sen_2, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_e1_token_idx, batch_e2_token_idx = [], []
        for sen_1, sen_2, event_idx in zip(batch_sen_1, batch_sen_2, batch_event_idx):
            e1_char_start, e1_char_end, e2_char_start, e2_char_end = event_idx
            encoding = tokenizer(sen_1, sen_2, max_length=args.max_seq_length, truncation=True)
            e1_start = encoding.char_to_token(e1_char_start, sequence_index=0)
            if not e1_start:
                e1_start = encoding.char_to_token(e1_char_start + 1, sequence_index=0)
            e1_end = encoding.char_to_token(e1_char_end, sequence_index=0)
            e2_start = encoding.char_to_token(e2_char_start, sequence_index=1)
            if not e2_start:
                e2_start = encoding.char_to_token(e2_char_start + 1, sequence_index=1)
            e2_end = encoding.char_to_token(e2_char_end, sequence_index=1)
            assert e1_start and e1_end and e2_start and e2_end
            batch_e1_token_idx.append([[e1_start, e1_end]])
            batch_e2_token_idx.append([[e2_start, e2_end]])
        for b_idx in range(len(batch_label)):
            e1_start, e1_end = batch_e1_token_idx[b_idx][0]
            e2_start, e2_end = batch_e2_token_idx[b_idx][0]
            batch_inputs_with_mask['input_ids'][b_idx][e1_start:e1_end+1] = tokenizer.mask_token_id
            batch_inputs_with_mask['input_ids'][b_idx][e2_start:e2_end+1] = tokenizer.mask_token_id
        return {
            'batch_inputs': batch_inputs, 
            'batch_inputs_with_mask': batch_inputs_with_mask, 
            'batch_e1_idx': batch_e1_token_idx, 
            'batch_e2_idx': batch_e2_token_idx, 
            'batch_e1_dists': batch_e1_dists, 
            'batch_e2_dists': batch_e2_dists, 
            'labels': batch_label, 
            'subtypes': batch_subtypes
        }

    if collote_fn_type == 'normal':
        select_collote_fn = collote_fn
    elif collote_fn_type == 'with_mask':
        select_collote_fn = collote_fn_with_mask
    elif collote_fn_type == 'with_dist':
        select_collote_fn = collote_fn_with_dist
    elif collote_fn_type == 'with_mask_dist':
        select_collote_fn = collote_fn_with_mask_dist

    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle, 
        collate_fn=select_collote_fn
    )
