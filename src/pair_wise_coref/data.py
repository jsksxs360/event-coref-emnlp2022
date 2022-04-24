import torch
from torch.utils.data import Dataset, DataLoader
import json

NO_CUTE = ['bert', 'spanbert']

class KBPCorefPair(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
    
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
                sentences = sample['sentences']
                clusters = sample['clusters']
                events = [
                    (
                        e['start'], 
                        e['sent_start'], 
                        e['sent_start'] + len(e['trigger']) - 1, 
                        sentences[e['sent_idx']]['text'], 
                        self._get_event_cluster_id(e['event_id'], clusters)
                    )
                    for e in sample['events']
                ]
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        e1_offset, e1_start, e1_end, e1_sen, e1_cluster = events[i]
                        e2_offset, e2_start, e2_end, e2_sen, e2_cluster = events[j]
                        Data.append({
                            'id': sample['doc_id'], 
                            'e1_offset': e1_offset, 
                            'e1_sen': e1_sen, 
                            'e1_start': e1_start, 
                            'e1_end': e1_end,
                            'e2_offset': e2_offset, 
                            'e2_sen': e2_sen, 
                            'e2_start': e2_start, 
                            'e2_end': e2_end, 
                            'label': 1 if e1_cluster == e2_cluster else 0
                        })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def cut_sent(sent, e_char_start, e_char_end, max_length):
    before = ' '.join([c for c in sent[:e_char_start].split(' ') if c != ''][-max_length:]).strip()
    trigger = sent[e_char_start:e_char_end+1]
    after = ' '.join([c for c in sent[e_char_end+1:].split(' ') if c != ''][:max_length]).strip()
    return before + ' ' + trigger + ' ' + after, len(before) + 1, len(before) + len(trigger)

def get_dataLoader(args, dataset, tokenizer, batch_size=None, shuffle=False, return_mask_inputs=False):

    def collote_fn(batch_samples):
        batch_sen_1, batch_sen_2, batch_event_idx, batch_label  = [], [], [], []
        for sample in batch_samples:
            if args.model_type in NO_CUTE:
                sen_1, e1_char_start, e1_char_end = sample['e1_sen'], sample['e1_start'], sample['e1_end']
                sen_2, e2_char_start, e2_char_end = sample['e2_sen'], sample['e2_start'], sample['e2_end']
            else:
                max_length = (args.max_seq_length - 50) // 4
                sen_1, e1_char_start, e1_char_end = cut_sent(sample['e1_sen'], sample['e1_start'], sample['e1_end'], max_length)
                sen_2, e2_char_start, e2_char_end = cut_sent(sample['e2_sen'], sample['e2_start'], sample['e2_end'], max_length)
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
        batch_inputs['batch_e1_idx'] = torch.tensor(batch_e1_token_idx)
        batch_inputs['batch_e2_idx'] = torch.tensor(batch_e2_token_idx)
        batch_inputs['labels'] = torch.tensor(batch_label)
        return batch_inputs
    
    def collote_fn_with_mask(batch_samples):
        batch_sen_1, batch_sen_2, batch_event_idx, batch_label  = [], [], [], []
        for sample in batch_samples:
            if args.model_type in NO_CUTE:
                sen_1, e1_char_start, e1_char_end = sample['e1_sen'], sample['e1_start'], sample['e1_end']
                sen_2, e2_char_start, e2_char_end = sample['e2_sen'], sample['e2_start'], sample['e2_end']
            else:
                max_length = (args.max_seq_length - 50) // 4
                sen_1, e1_char_start, e1_char_end = cut_sent(sample['e1_sen'], sample['e1_start'], sample['e1_end'], max_length)
                sen_2, e2_char_start, e2_char_end = cut_sent(sample['e2_sen'], sample['e2_start'], sample['e2_end'], max_length)
            batch_sen_1.append(sen_1)
            batch_sen_2.append(sen_2)
            batch_event_idx.append(
                (e1_char_start, e1_char_end, e2_char_start, e2_char_end)
            )
            batch_label.append(sample['label'])
        batch_inputs = tokenizer(
            batch_sen_1, batch_sen_2, 
            max_length=args.max_seq_length, padding=True, truncation=True, return_tensors="pt"
        )
        batch_inputs_with_mask = tokenizer(
            batch_sen_1, batch_sen_2, 
            max_length=args.max_seq_length, padding=True, truncation=True, return_tensors="pt"
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
            'batch_e1_idx': torch.tensor(batch_e1_token_idx), 
            'batch_e2_idx': torch.tensor(batch_e2_token_idx), 
            'labels': torch.tensor(batch_label)
        }
    
    return DataLoader(dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, 
                      collate_fn=collote_fn_with_mask if return_mask_inputs else collote_fn)
