from torch.utils.data import Dataset, DataLoader
import json

NO_CUTE = ['bert', 'spanbert']

SUBTYPES = [ # 18 subtypes
    'artifact', 'transferownership', 'transaction', 'broadcast', 'contact', 'demonstrate', \
    'injure', 'transfermoney', 'transportartifact', 'attack', 'meet', 'elect', \
    'endposition', 'correspondence', 'arrestjail', 'startposition', 'transportperson', 'die'
]
id2subtype = {idx: c for idx, c in enumerate(SUBTYPES, start=1)}
subtype2id = {v: k for k, v in id2subtype.items()}

class KBPCoref(Dataset):
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
                event_mentions, mention_char_pos = [], []
                for e in sample['events']:
                    event_mentions.append(sentences[e['sent_idx']]['text'])
                    mention_char_pos.append([
                        e['sent_start'], e['sent_start'] + len(e['trigger']) - 1
                    ])
                Data.append({
                    'id': sample['doc_id'], 
                    'document': sample['document'], 
                    'events': events, # [{event_id, char_start, char_end, trigger, subtype, cluster_id}]
                    'event_mentions': event_mentions, 
                    'mention_char_pos': mention_char_pos # [event_mention_start, event_mention_end]
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

def get_dataLoader(args, dataset, tokenizer, mention_tokenizer=None, batch_size=None, shuffle=False, collote_fn_type='normal'):

    assert collote_fn_type in ['normal', 'with_mention', 'with_mask_subtype']

    def collote_fn(batch_samples):
        batch_sentences, batch_events  = [], []
        for sample in batch_samples:
            batch_sentences.append(sample['document'])
            batch_events.append(sample['events'])
        batch_inputs = tokenizer(
            batch_sentences, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_filtered_events = []
        batch_filtered_event_cluster_id = []
        for b_idx, sentence in enumerate(batch_sentences):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            filtered_events = []
            filtered_event_cluster_id = []
            for e in batch_events[b_idx]:
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
        batch_inputs['batch_events'] = batch_filtered_events
        batch_inputs['batch_event_cluster_ids'] = batch_filtered_event_cluster_id
        return batch_inputs
    
    def collote_fn_with_mention(batch_samples):
        batch_sentences, batch_events, batch_mentions, batch_mention_pos = [], [], [], []
        for sample in batch_samples:
            batch_sentences.append(sample['document'])
            batch_events.append(sample['events'])
            batch_mentions.append(sample['event_mentions'])
            batch_mention_pos.append(sample['mention_char_pos'])
        batch_inputs = tokenizer(
            batch_sentences, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_filtered_events = []
        batch_filtered_mention_inputs = []
        batch_filtered_mention_events = []
        batch_filtered_event_cluster_id = []
        for b_idx, sentence in enumerate(batch_sentences):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            filtered_events = []
            filtered_event_mentions = []
            filtered_mention_events = []
            filtered_event_cluster_id = []
            for event, mention, mention_pos in zip(
                batch_events[b_idx], batch_mentions[b_idx], batch_mention_pos[b_idx]
            ):
                token_start = encoding.char_to_token(event['char_start'])
                if not token_start:
                    token_start = encoding.char_to_token(event['char_start'] + 1)
                token_end = encoding.char_to_token(event['char_end'])
                if not token_start or not token_end:
                    continue
                # cut long mentions for Roberta-like model
                mention_char_start, mention_char_end = mention_pos
                if args.mention_encoder_type not in NO_CUTE:
                    max_length = (args.max_mention_length - 50) // 2
                    mention, mention_char_start, mention_char_end = cut_sent(
                        mention, mention_char_start, mention_char_end, max_length
                    )
                mention_encoding = mention_tokenizer(mention, max_length=args.max_mention_length, truncation=True)
                mention_token_start = mention_encoding.char_to_token(mention_char_start)
                if not mention_token_start:
                    mention_token_start = mention_encoding.char_to_token(mention_char_start + 1)
                mention_token_end = mention_encoding.char_to_token(mention_char_end)
                assert mention_token_start and mention_token_end
                filtered_events.append([token_start, token_end])
                filtered_event_mentions.append(mention)
                filtered_mention_events.append([mention_token_start, mention_token_end])
                filtered_event_cluster_id.append(event['cluster_id'])
            batch_filtered_events.append(filtered_events)
            batch_filtered_mention_inputs.append(
                mention_tokenizer(
                    filtered_event_mentions, 
                    max_length=args.max_mention_length, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
            )
            batch_filtered_mention_events.append(filtered_mention_events)
            batch_filtered_event_cluster_id.append(filtered_event_cluster_id)
        return {
            'batch_inputs': batch_inputs, 
            'batch_events': batch_filtered_events, 
            'batch_mention_inputs': batch_filtered_mention_inputs, 
            'batch_mention_events': batch_filtered_mention_events, 
            'batch_event_cluster_ids': batch_filtered_event_cluster_id
        }
    
    def collote_fn_with_mask_subtyep(batch_samples):
        batch_sentences, batch_events = [], []
        batch_mentions, batch_mention_pos = [], []
        for sample in batch_samples:
            batch_sentences.append(sample['document'])
            batch_events.append(sample['events'])
            batch_mentions.append(sample['event_mentions'])
            batch_mention_pos.append(sample['mention_char_pos'])
        batch_inputs = tokenizer(
            batch_sentences, 
            max_length=args.max_seq_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        batch_filtered_events = []
        batch_filtered_mention_inputs_with_mask = []
        batch_filtered_mention_events = []
        batch_filtered_event_cluster_id = []
        batch_filtered_event_subtypes = []
        for b_idx, sentence in enumerate(batch_sentences):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            filtered_events = []
            filtered_event_mentions = []
            filtered_mention_events = []
            filtered_event_cluster_id = []
            filtered_event_subtypes = []
            for event, mention, mention_pos in zip(
                batch_events[b_idx], batch_mentions[b_idx], batch_mention_pos[b_idx]
            ):
                token_start = encoding.char_to_token(event['char_start'])
                if not token_start:
                    token_start = encoding.char_to_token(event['char_start'] + 1)
                token_end = encoding.char_to_token(event['char_end'])
                if not token_start or not token_end:
                    continue
                # cut long mention for Roberta-like model
                mention_char_start, mention_char_end = mention_pos
                if args.mention_encoder_type not in NO_CUTE:
                    max_length = (args.max_mention_length - 50) // 2
                    mention, mention_char_start, mention_char_end = cut_sent(
                        mention, mention_char_start, mention_char_end, max_length
                    )
                mention_encoding = mention_tokenizer(mention, max_length=args.max_mention_length, truncation=True)
                mention_token_start = mention_encoding.char_to_token(mention_char_start)
                if not mention_token_start:
                    mention_token_start = mention_encoding.char_to_token(mention_char_start + 1)
                mention_token_end = mention_encoding.char_to_token(mention_char_end)
                assert mention_token_start and mention_token_end
                filtered_events.append([token_start, token_end])
                filtered_event_mentions.append(mention)
                filtered_mention_events.append([mention_token_start, mention_token_end])
                filtered_event_cluster_id.append(event['cluster_id'])
                filtered_event_subtypes.append(event['subtype'])
            batch_filtered_events.append(filtered_events)
            batch_filtered_mention_inputs_with_mask.append(
                mention_tokenizer(
                    filtered_event_mentions, 
                    max_length=args.max_mention_length, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
            )
            batch_filtered_mention_events.append(filtered_mention_events)
            batch_filtered_event_cluster_id.append(filtered_event_cluster_id)
            batch_filtered_event_subtypes.append(filtered_event_subtypes)
        for b_idx in range(len(batch_filtered_event_subtypes)):
            for e_idx, (e_start, e_end) in enumerate(batch_filtered_mention_events[b_idx]):
                batch_filtered_mention_inputs_with_mask[b_idx]['input_ids'][e_idx][e_start:e_end+1] = mention_tokenizer.mask_token_id
        return {
            'batch_inputs': batch_inputs, 
            'batch_events': batch_filtered_events, 
            'batch_mention_inputs_with_mask': batch_filtered_mention_inputs_with_mask, 
            'batch_mention_events': batch_filtered_mention_events, 
            'batch_event_cluster_ids': batch_filtered_event_cluster_id, 
            'batch_event_subtypes': batch_filtered_event_subtypes
        }

    if collote_fn_type == 'normal':
        select_collote_fn = collote_fn
    elif collote_fn_type == 'with_mention':
        assert mention_tokenizer is not None
        select_collote_fn = collote_fn_with_mention
    elif collote_fn_type == 'with_mask_subtype':
        assert mention_tokenizer is not None
        select_collote_fn = collote_fn_with_mask_subtyep
    return DataLoader(
        dataset, 
        batch_size=(batch_size if batch_size else args.batch_size), 
        shuffle=shuffle, 
        collate_fn=select_collote_fn
    )
