import logging
from tqdm.auto import tqdm
import json
from collections import namedtuple, defaultdict
import torch
from transformers import AdamW, get_scheduler
from transformers import AutoConfig, AutoTokenizer
import numpy as np
from sklearn.metrics import classification_report
import os
import sys
sys.path.append('../../')
from src.tools import seed_everything, NpEncoder
from src.global_event_coref.arg import parse_args
from src.global_event_coref.data import KBPCoref, get_dataLoader, cut_sent, SUBTYPES, vocab, VOCAB_SIZE
from src.global_event_coref.modeling import LongformerSoftmaxForECwithMaskTopic

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")
Sentence = namedtuple("Sentence", ["start", "text"])

def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k in ['batch_events', 'batch_mention_events', 'batch_event_cluster_ids', 'batch_event_subtypes']:
            new_batch_data[k] = v
        elif k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        elif k == 'batch_event_dists':
            new_batch_data[k] = [
                torch.tensor(event_dists, dtype=torch.float32).to(args.device) 
                for event_dists in v
            ]
        elif k == 'batch_mention_inputs_with_mask':
            new_batch_data[k] = [
                {k_: v_.to(args.device) for k_, v_ in inputs.items()} 
                for inputs in v
            ]
        else:
            raise ValueError(f'Unknown batch data key: {k}')
    return new_batch_data

def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(dataloader)
    
    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(args, batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        if loss:
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            lr_scheduler.step()

        total_loss += loss.item() if loss else 0.
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(args, dataloader, model):
    true_labels, true_predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            _, logits, masks, labels = outputs

            predictions = logits.argmax(dim=-1).cpu().numpy() # [batch, event_pair_num]
            y = labels.cpu().numpy()
            lens = np.sum(masks.cpu().numpy(), axis=-1)
            true_labels += [
                int(l) for label, seq_len in zip(y, lens) for idx, l in enumerate(label) if idx < seq_len
            ]
            true_predictions += [
                int(p) for pred, seq_len in zip(predictions, lens) for idx, p in enumerate(pred) if idx < seq_len
            ]
    return classification_report(true_labels, true_predictions, output_dict=True)

def train(args, train_dataset, dev_dataset, model, tokenizer, mention_tokenizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, mention_tokenizer, shuffle=True, collote_fn_type='with_mask_dist')
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, mention_tokenizer, shuffle=False, collote_fn_type='with_mask_dist')
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))

    total_loss = 0.
    best_f1 = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(args, dev_dataloader, model)
        dev_p, dev_r, dev_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
        logger.info(f'Dev: P - {(100*dev_p):0.4f} R - {(100*dev_r):0.4f} F1 - {(100*dev_f1):0.4f}')
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_f1_{(100*dev_f1):0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
        elif 100 * dev_p > 69 and 100 * dev_r > 69:
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_f1_{(100*dev_f1):0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
        with open(os.path.join(args.output_dir, 'dev_metrics.txt'), 'at') as f:
            f.write(f'epoch_{epoch+1}\n' + json.dumps(metrics, cls=NpEncoder) + '\n\n')
    logger.info("Done!")

def predict(args, document:str, events:list, mentions:list, mention_pos:list, event_dists:list, model, tokenizer, mention_tokenizer):
    assert len(events) == len(mentions) == len(mention_pos) == len(event_dists)
    inputs = tokenizer(
        document, 
        max_length=args.max_seq_length, 
        truncation=True, 
        return_tensors="pt"
    )
    filtered_events = []
    new_events = []
    filtered_mentions = []
    filtered_mention_events = []
    filtered_dists = []
    for event, mention, mention_pos, event_dist in zip(events, mentions, mention_pos, event_dists):
        char_start, char_end = event
        token_start = inputs.char_to_token(char_start)
        if not token_start:
            token_start = inputs.char_to_token(char_start + 1)
        token_end = inputs.char_to_token(char_end)
        if not token_start or not token_end:
            continue
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
        filtered_events.append([token_start, token_end])
        new_events.append(event)
        filtered_mentions.append(mention)
        filtered_mention_events.append([mention_token_start, mention_token_end])
        filtered_dists.append(event_dist)
    filtered_mention_inputs_with_mask = mention_tokenizer(
        filtered_mentions, 
        max_length=args.max_mention_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    for e_idx, (e_start, e_end) in enumerate(filtered_mention_events):
        filtered_mention_inputs_with_mask['input_ids'][e_idx][e_start:e_end+1] = mention_tokenizer.mask_token_id
    if not new_events:
        return [], [], []
    inputs = {
        'batch_inputs': inputs, 
        'batch_events': [filtered_events], 
        'batch_mention_inputs_with_mask': [filtered_mention_inputs_with_mask], 
        'batch_mention_events': [filtered_mention_events], 
        'batch_event_dists': [np.asarray(filtered_dists)]
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
    predictions = logits.argmax(dim=-1)[0].cpu().numpy().tolist()
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
    probabilities = [probabilities[idx][pred] for idx, pred in enumerate(predictions)]
    if len(new_events) > 1:
        assert len(predictions) == len(new_events) * (len(new_events) - 1) / 2
    return new_events, predictions, probabilities

def test(args, test_dataset, model, tokenizer, mention_tokenizer, save_weights:list):
    test_dataloader = get_dataLoader(
        args, test_dataset, tokenizer, mention_tokenizer, batch_size=1, shuffle=False, 
        collote_fn_type='with_mask_dist'
    )
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'at') as f:
            f.write(f'{save_weight}\n{json.dumps(metrics, cls=NpEncoder)}\n\n')

def get_event_sent(e_start, e_end, sents):
    for sent in sents:
        sent_end = sent.start + len(sent.text) - 1
        if e_start >= sent.start and e_end <= sent_end:
            return sent.text, e_start - sent.start,  e_end - sent.start
    return None, None, None

def get_event_dist(e_start, e_end, sents):
    for s_idx, sent in enumerate(sents):
        sent_end = sent.start + len(sent.text) - 1
        if e_start >= sent.start and e_end <= sent_end:
            before = sents[s_idx - 1].text if s_idx > 0 else ''
            after = sents[s_idx + 1].text if s_idx < len(sents) - 1 else ''
            event_mention = before + (' ' if len(before) > 0 else '') + sent.text + ' ' + after
            event_mention = event_mention.lower()
            return [1 if w in event_mention else 0 for w in vocab]
    return None

if __name__ == '__main__':
    args = parse_args()
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} & {args.mention_encoder_type}...')
    logger.info(f'using model {"with" if args.add_contrastive_loss else "without"} Contrastive loss')
    main_config = AutoConfig.from_pretrained(args.model_checkpoint, cache_dir=args.cache_dir)
    encoder_config = AutoConfig.from_pretrained(args.mention_encoder_checkpoint, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, cache_dir=args.cache_dir)
    mention_tokenizer = AutoTokenizer.from_pretrained(args.mention_encoder_checkpoint, cache_dir=args.cache_dir)
    args.num_labels = 2
    args.num_subtypes = len(SUBTYPES) + 1
    args.dist_dim = VOCAB_SIZE
    model = LongformerSoftmaxForECwithMaskTopic.from_pretrained(
        args.model_checkpoint,
        config=main_config,
        encoder_config=encoder_config, 
        cache_dir=args.cache_dir,
        args=args
    ).to(args.device)
    # Training
    if args.do_train:
        train_dataset = KBPCoref(args.train_file, include_mention_context=args.include_mention_context)
        dev_dataset = KBPCoref(args.dev_file, include_mention_context=args.include_mention_context)
        train(args, train_dataset, dev_dataset, model, tokenizer, mention_tokenizer)
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = KBPCoref(args.test_file, include_mention_context=args.include_mention_context)
        test(args, test_dataset, model, tokenizer, mention_tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        kbp_sent_dic = defaultdict(list) # {filename: [Sentence]}
        with open(os.path.join('../../data/kbp_sent.txt'), 'rt', encoding='utf-8') as sents:
            for line in sents:
                doc_id, start, text = line.strip().split('\t')
                kbp_sent_dic[doc_id].append(Sentence(int(start), text))

        pred_event_file = 'epoch_3_dev_f1_57.9994_weights.bin_test_pred_events.json'
        # pred_event_file = 'test_filtered.json'

        for best_save_weight in save_weights:
            logger.info(f'loading weights from {best_save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, best_save_weight)))
            logger.info(f'predicting coref labels of {best_save_weight}...')
            
            results = []
            model.eval()
            with open(os.path.join(args.output_dir, pred_event_file), 'rt' , encoding='utf-8') as f_in:
                for line in tqdm(f_in.readlines()):
                    sample = json.loads(line.strip())
                    events_from_file = sample['events'] if pred_event_file == 'test_filtered.json' else sample['pred_label']
                    events = [
                        [event['start'], event['start'] + len(event['trigger']) - 1] 
                        for event in events_from_file
                    ]
                    sents = kbp_sent_dic[sample['doc_id']]
                    mentions, mention_pos, event_dists = [], [], []
                    for event in events_from_file:
                        e_sent, e_new_start, e_new_end = get_event_sent(event['start'], event['start'] + len(event['trigger']) - 1, sents)
                        assert e_sent is not None and e_sent[e_new_start:e_new_end+1] == event['trigger']
                        mentions.append(e_sent)
                        mention_pos.append([e_new_start, e_new_end])
                        e_dist = get_event_dist(event['start'], event['start'] + len(event['trigger']) - 1, sents)
                        assert e_dist is not None
                        event_dists.append(e_dist)
                    new_events, predictions, probabilities = predict(
                        args, sample['document'], events, mentions, mention_pos, event_dists, model, tokenizer, mention_tokenizer
                    )
                    results.append({
                        "doc_id": sample['doc_id'], 
                        "document": sample['document'], 
                        "events": [
                            {
                                'start': char_start, 
                                'end': char_end, 
                                'trigger': sample['document'][char_start:char_end+1]
                            } for char_start, char_end in new_events
                        ], 
                        "pred_label": predictions, 
                        "pred_prob": probabilities
                    })
            save_name = '_gold_test_pred_corefs.json' if pred_event_file == 'test_filtered.json' else '_test_pred_corefs.json'
            with open(os.path.join(args.output_dir, best_save_weight + save_name), 'wt', encoding='utf-8') as f:
                for exapmle_result in results:
                    f.write(json.dumps(exapmle_result) + '\n')
