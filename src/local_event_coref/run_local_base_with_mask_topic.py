import os
import logging
import torch
import json
from collections import namedtuple, defaultdict
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
from sklearn.metrics import classification_report
import sys
sys.path.append('../../')
from src.tools import seed_everything, NpEncoder
from src.local_event_coref.arg import parse_args
from src.local_event_coref.data import KBPCorefPair, get_dataLoader, SUBTYPES, vocab, VOCAB_SIZE
from src.local_event_coref.modeling import BertForPairwiseECwithMaskTopic, RobertaForPairwiseECwithMaskTopic

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")
Sentence = namedtuple("Sentence", ["start", "text"])

MODEL_CLASSES = {
    'bert': BertForPairwiseECwithMaskTopic,
    'spanbert': BertForPairwiseECwithMaskTopic, 
    'roberta': RobertaForPairwiseECwithMaskTopic
}

def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k in ['batch_inputs', 'batch_inputs_with_mask']:
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        elif k in ['batch_e1_dists', 'batch_e2_dists']:
            new_batch_data[k] = torch.tensor(v, dtype=torch.float32).to(args.device) 
        else:
            new_batch_data[k] = torch.tensor(v).to(args.device)
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

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
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
            logits = outputs[1]

            predictions = logits.argmax(dim=-1).cpu().numpy().tolist()
            labels = batch_data['labels'].cpu().numpy()
            true_predictions += predictions
            true_labels += [int(label) for label in labels]
    return classification_report(true_labels, true_predictions, output_dict=True)

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, shuffle=True, collote_fn_type='with_mask_dist')
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, shuffle=False, collote_fn_type='with_mask_dist')
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
        with open(os.path.join(args.output_dir, 'dev_metrics.txt'), 'at') as f:
            f.write(f'epoch_{epoch+1}\n' + json.dumps(metrics, cls=NpEncoder) + '\n\n')
    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights:list):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, batch_size=1, shuffle=False, collote_fn_type='with_mask_dist')
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'at') as f:
            f.write(f'{save_weight}\n{json.dumps(metrics, cls=NpEncoder)}\n\n')

def predict(args, sent_1, sent_2, e1_char_start, e1_char_end, e2_char_start, e2_char_end, e1_dist, e2_dist, model, tokenizer):

    def _cut_sent(sent, e_char_start, e_char_end, max_length):
        before = ' '.join([c for c in sent[:e_char_start].split(' ') if c != ''][-max_length:]).strip()
        trigger = sent[e_char_start:e_char_end+1]
        after = ' '.join([c for c in sent[e_char_end+1:].split(' ') if c != ''][:max_length]).strip()
        new_sent, new_char_start, new_char_end = before + ' ' + trigger + ' ' + after, len(before) + 1, len(before) + len(trigger)
        assert new_sent[new_char_start:new_char_end+1] == trigger
        return new_sent, new_char_start, new_char_end

    max_mention_length = (args.max_seq_length - 50) // 4
    sent_1, e1_char_start, e1_char_end = _cut_sent(sent_1, e1_char_start, e1_char_end, max_mention_length)
    sent_2, e2_char_start, e2_char_end = _cut_sent(sent_2, e2_char_start, e2_char_end, max_mention_length)
    inputs = tokenizer(
        sent_1, 
        sent_2, 
        max_length=args.max_seq_length, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs_with_mask = tokenizer(
        sent_1, 
        sent_2, 
        max_length=args.max_seq_length, 
        truncation=True, 
        return_tensors="pt"
    )
    e1_token_start = inputs.char_to_token(e1_char_start, sequence_index=0)
    if not e1_token_start:
        e1_token_start = inputs.char_to_token(e1_char_start + 1, sequence_index=0)
    e1_token_end = inputs.char_to_token(e1_char_end, sequence_index=0)
    e2_token_start = inputs.char_to_token(e2_char_start, sequence_index=1)
    if not e2_token_start:
        e2_token_start = inputs.char_to_token(e2_char_start + 1, sequence_index=1)
    e2_token_end = inputs.char_to_token(e2_char_end, sequence_index=1)
    assert e1_token_start and e1_token_end and e2_token_start and e2_token_end
    inputs_with_mask['input_ids'][0][e1_token_start:e1_token_end+1] = tokenizer.mask_token_id
    inputs_with_mask['input_ids'][0][e2_token_start:e2_token_end+1] = tokenizer.mask_token_id
    inputs = {
        'batch_inputs': inputs, 
        'batch_inputs_with_mask': inputs_with_mask, 
        'batch_e1_idx': [[[e1_token_start, e1_token_end]]], 
        'batch_e2_idx': [[[e2_token_start, e2_token_end]]], 
        'batch_e1_dists': [e1_dist], 
        'batch_e2_dists': [e2_dist]
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
    pred = int(logits.argmax(dim=-1)[0].cpu().numpy())
    prob = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
    return pred, prob[pred]

def get_event_sent(e_start, e_end, sents):
    for sent in sents:
        sent_end = sent.start + len(sent.text) - 1
        if e_start >= sent.start and e_end <= sent_end:
            return sent.text, e_start - sent.start, e_end - sent.start
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
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    config = AutoConfig.from_pretrained(args.model_checkpoint, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, cache_dir=args.cache_dir)
    args.num_labels = 2
    args.num_subtypes = len(SUBTYPES) + 1
    args.dist_dim = VOCAB_SIZE
    if args.do_test or args.do_predict:
        args.batch_size = 1
    model = MODEL_CLASSES[args.model_type].from_pretrained(
        args.model_checkpoint,
        config=config,
        cache_dir=args.cache_dir, 
        args=args
    ).to(args.device)
    # Training
    save_weights = []
    if args.do_train:
        train_dataset = KBPCorefPair(args.train_file)
        dev_dataset = KBPCorefPair(args.dev_file)
        save_weights = train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = KBPCorefPair(args.test_file)
        test(args, test_dataset, model, tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        kbp_sent_dic = defaultdict(list) # {filename: [Sentence]}
        with open(os.path.join('../../data/kbp_sent.txt'), 'rt', encoding='utf-8') as sents:
            for line in sents:
                doc_id, start, text = line.strip().split('\t')
                kbp_sent_dic[doc_id].append(Sentence(int(start), text))

        pred_event_file = 'epoch_3_dev_f1_57.9994_weights.bin_test_pred_events.json'

        for best_save_weight in save_weights:
            logger.info(f'loading weights from {best_save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, best_save_weight)))
            logger.info(f'predicting coref labels of {best_save_weight}...')
            
            results = []
            model.eval()
            pred_event_filepath = os.path.join(args.output_dir, pred_event_file)
            with open(pred_event_filepath, 'rt' , encoding='utf-8') as f_in:
                for line in tqdm(f_in.readlines()):
                    sample = json.loads(line.strip())
                    events = [
                        (event['start'], event['start'] + len(event['trigger']) - 1, event['trigger'])
                        for event in sample['pred_label']
                    ]
                    sents = kbp_sent_dic[sample['doc_id']]
                    new_events = []
                    for e_start, e_end, e_trigger in events:
                        e_sent, e_new_start, e_new_end = get_event_sent(e_start, e_end, sents)
                        assert e_sent is not None and e_sent[e_new_start:e_new_end+1] == e_trigger
                        e_dist = get_event_dist(e_start, e_end, sents)
                        new_events.append((e_new_start, e_new_end, e_sent, e_dist))
                    predictions, probabilities = [], []
                    for i in range(len(new_events) - 1):
                        for j in range(i + 1, len(new_events)):
                            e1_char_start, e1_char_end, sent_1, e1_dist = new_events[i]
                            e2_char_start, e2_char_end, sent_2, e2_dist = new_events[j]
                            pred, prob = predict(args, 
                                sent_1, sent_2, 
                                e1_char_start, e1_char_end, 
                                e2_char_start, e2_char_end, 
                                e1_dist, e2_dist, 
                                model, tokenizer
                            )
                            predictions.append(pred)
                            probabilities.append(prob)
                    results.append({
                        "doc_id": sample['doc_id'], 
                        "document": sample['document'], 
                        "events": [
                            {
                                'start': char_start, 
                                'end': char_end, 
                                'trigger': trigger
                            } for char_start, char_end, trigger in events
                        ], 
                        "pred_label": predictions, 
                        "pred_prob": probabilities
                    })
            with open(os.path.join(args.output_dir, best_save_weight + '_test_pred_corefs.json'), 'wt', encoding='utf-8') as f:
                for exapmle_result in results:
                    f.write(json.dumps(exapmle_result) + '\n')
