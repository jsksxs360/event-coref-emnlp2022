from torch.utils.data import Dataset, DataLoader
import json

NO_CUTE = ['bert', 'spanbert']

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
                events = [
                    [e['event_id'], e['start'], e['start']+len(e['trigger'])-1, e['trigger']] 
                    for e in sample['events']
                ]
                clusters = sample['clusters']
                for event in events:
                    event.append(self._get_event_cluster_id(event[0], clusters))
                sentences = sample['sentences']
                event_mentions, mention_char_pos = [], []
                for e in sample['events']:
                    event_mentions.append(sentences[e['sent_idx']]['text'])
                    mention_char_pos.append([
                        e['sent_start'], e['sent_start']+len(e['trigger'])-1
                    ])
                Data.append({
                    'id': sample['doc_id'], 
                    'document': sample['document'], 
                    'events': events, # [event_id, char_start, char_end, trigger, cluster_id]
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

def get_dataLoader(args, dataset, tokenizer, mention_tokenizer=None, batch_size=None, shuffle=False):
    
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
        for s_idx, sentence in enumerate(batch_sentences):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            filtered_events = []
            filtered_event_cluster_id = []
            for _, char_start, char_end, _, cluster_id in batch_events[s_idx]:
                token_start = encoding.char_to_token(char_start)
                if not token_start:
                    token_start = encoding.char_to_token(char_start + 1)
                token_end = encoding.char_to_token(char_end)
                if not token_start or not token_end:
                    continue
                filtered_events.append([token_start, token_end])
                filtered_event_cluster_id.append(cluster_id)
            batch_filtered_events.append(filtered_events)
            batch_filtered_event_cluster_id.append(filtered_event_cluster_id)
        batch_inputs['batch_events'] = batch_filtered_events
        batch_inputs['batch_event_cluster_ids'] = batch_filtered_event_cluster_id
        return batch_inputs
    
    def collote_fn_with_mention(batch_samples):
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
        batch_filtered_mention_inputs = []
        batch_filtered_mention_events = []
        batch_filtered_event_cluster_id = []
        for b_idx, sentence in enumerate(batch_sentences):
            encoding = tokenizer(sentence, max_length=args.max_seq_length, truncation=True)
            filtered_events = []
            filtered_mention_events = []
            filtered_event_cluster_id = []
            for event, mention, mention_pos in zip(batch_events[b_idx], batch_mentions[b_idx], batch_mention_pos[b_idx]):
                _, char_start, char_end, _, cluster_id = event
                token_start = encoding.char_to_token(char_start)
                if not token_start:
                    token_start = encoding.char_to_token(char_start + 1)
                token_end = encoding.char_to_token(char_end)
                if not token_start or not token_end:
                    continue
                filtered_events.append([token_start, token_end])
                filtered_event_cluster_id.append(cluster_id)
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
                filtered_mention_events.append([mention_token_start, mention_token_end])
            batch_filtered_events.append(filtered_events)
            batch_filtered_mention_inputs.append(
                mention_tokenizer(
                    batch_mentions[b_idx], 
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
    
    return DataLoader(dataset, batch_size=(batch_size if batch_size else args.batch_size), shuffle=shuffle, 
                      collate_fn=collote_fn_with_mention if mention_tokenizer else collote_fn)
