import os
import logging
import torch
import json
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
import numpy as np
from seqeval.metrics import classification_report as td_cls_report
from seqeval.scheme import IOB2
from sklearn.metrics import classification_report as ec_cls_report
import sys
sys.path.append('../../')
from src.tools import seed_everything, NpEncoder
from src.joint_model.arg import parse_args
from src.joint_model.data import KBPCoref, get_dataLoader, SUBTYPES
from src.joint_model.modeling import LongformerSoftmaxForEC

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

def to_device(args, batch_data):
    new_batch_data = {}
    for k, v in batch_data.items():
        if k in ['batch_events', 'batch_event_cluster_ids']:
            new_batch_data[k] = v
        elif k == 'batch_inputs':
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        elif k == 'batch_td_labels':
            new_batch_data[k] = v.to(args.device)
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
    true_td_labels, true_td_predictions = [], []
    true_ec_labels, true_ec_predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            _, td_logits, coref_logits, td_masks, td_labels, coref_masks, coref_labels = outputs
            # trigger detection
            td_predictions = td_logits.argmax(dim=-1).cpu().numpy() # [batch, seq]
            td_labels = td_labels.cpu().numpy()
            td_lens = np.sum(td_masks.cpu().numpy(), axis=-1)
            true_td_labels += [
                [args.id2label[int(l)] for idx, l in enumerate(label) if idx > 0 and idx < seq_len - 1] 
                for label, seq_len in zip(td_labels, td_lens)
            ]
            true_td_predictions += [
                [args.id2label[int(p)] for idx, p in enumerate(prediction) if idx > 0 and idx < seq_len - 1]
                for prediction, seq_len in zip(td_predictions, td_lens)
            ]
            # event coreference
            ec_predictions = coref_logits.argmax(dim=-1).cpu().numpy() # [batch, event_pair_num]
            ec_y = coref_labels.cpu().numpy()
            ec_lens = np.sum(coref_masks.cpu().numpy(), axis=-1)
            true_ec_labels += [
                int(l) for label, seq_len in zip(ec_y, ec_lens) for idx, l in enumerate(label) if idx < seq_len
            ]
            true_ec_predictions += [
                int(p) for pred, seq_len in zip(ec_predictions, ec_lens) for idx, p in enumerate(pred) if idx < seq_len
            ]
    return (
        td_cls_report(true_td_labels, true_td_predictions, mode='strict', scheme=IOB2, output_dict=True), 
        ec_cls_report(true_ec_labels, true_ec_predictions, output_dict=True)
    )

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, shuffle=False)
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
    best_td_f1, best_ec_f1 = 0., 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n-------------------------------")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        td_metrics, ec_metrics = test_loop(args, dev_dataloader, model)
        td_micro_f1, td_macro_f1 = td_metrics['micro avg']['f1-score'], td_metrics['macro avg']['f1-score']
        td_dev_f1 = td_metrics['weighted avg']['f1-score']
        ec_dev_p, ec_dev_r, ec_dev_f1 = ec_metrics['1']['precision'], ec_metrics['1']['recall'], ec_metrics['1']['f1-score']
        logger.info(f'TD: micro_F1 - {(100*td_micro_f1):0.4f} macro_f1 - {(100*td_macro_f1):0.4f} weighted_f1 - {(100*td_dev_f1):0.4f}')
        logger.info(f'EC: P - {(100*ec_dev_p):0.4f} R - {(100*ec_dev_r):0.4f} F1 - {(100*ec_dev_f1):0.4f}')
        has_saved = False
        if td_dev_f1 > best_td_f1:
            best_td_f1 = td_dev_f1
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_tdf1_{(100*td_dev_f1):0.4f}_ecf1_{(100*ec_dev_f1):0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
            has_saved = True
        if ec_dev_f1 > best_ec_f1:
            best_ec_f1 = ec_dev_f1
            if not has_saved:
                logger.info(f'saving new weights to {args.output_dir}...\n')
                save_weight = f'epoch_{epoch+1}_dev_tdf1_{(100*td_dev_f1):0.4f}_ecf1_{(100*ec_dev_f1):0.4f}_weights.bin'
                torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
                has_saved = True
        elif 100*ec_dev_p > 69 and 100*ec_dev_r > 69:
            if not has_saved:
                logger.info(f'saving new weights to {args.output_dir}...\n')
                save_weight = f'epoch_{epoch+1}_dev_tdf1_{(100*td_dev_f1):0.4f}_ecf1_{(100*ec_dev_f1):0.4f}_weights.bin'
                torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
        with open(os.path.join(args.output_dir, 'dev_metrics.txt'), 'at') as f:
            f.write(
                f'epoch_{epoch+1}\n' + json.dumps(td_metrics, cls=NpEncoder) + 
                '\n\n' + json.dumps(ec_metrics, cls=NpEncoder) + '\n\n\n'
            )
    logger.info("Done!")

def predict_td(args, document:str, model, tokenizer):
    inputs = tokenizer(
        document, 
        max_length=args.max_seq_length, 
        truncation=True, 
        return_tensors="pt", 
        return_offsets_mapping=True
    )
    offsets = inputs.pop('offset_mapping').squeeze(0)
    inputs = {
        'batch_inputs': inputs
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
    predictions = logits.argmax(dim=-1)[0].cpu().numpy().tolist()

    pred_label = []
    idx = 1
    while idx < len(predictions) - 1:
        pred = predictions[idx]
        label = args.id2label[pred]
        if label != "O":
            label = label[2:] # Remove the B- or I-
            start, end = offsets[idx]
            all_scores = [probabilities[idx][pred]]
            # Grab all the tokens labeled with I-label
            while (
                idx + 1 < len(predictions) - 1 and 
                args.id2label[predictions[idx + 1]] == f"I-{label}"
            ):
                all_scores.append(probabilities[idx + 1][predictions[idx + 1]])
                _, end = offsets[idx + 1]
                idx += 1

            score = np.mean(all_scores).item()
            start, end = start.item(), end.item()
            word = document[start:end]
            pred_label.append({
                "trigger": word, 
                "start": start, 
                "subtype": label, 
                "score": score
            })
        idx += 1
    return pred_label

def predict_ec(args, document:str, events:list, model, tokenizer):
    '''
    # Args:
        - events: [
            [e_char_start, e_char_end], ...
        ], document[e1_char_start:e1_char_end + 1] = trigger1
    '''
    inputs = tokenizer(
        document, 
        max_length=args.max_seq_length, 
        truncation=True, 
        return_tensors="pt"
    )
    filtered_events = []
    new_events = []
    for event in events:
        char_start, char_end = event
        token_start = inputs.char_to_token(char_start)
        if not token_start:
            token_start = inputs.char_to_token(char_start + 1)
        token_end = inputs.char_to_token(char_end)
        if not token_start or not token_end:
            continue
        filtered_events.append([token_start, token_end])
        new_events.append(event)
    if not new_events:
        return [], [], []
    inputs = {
        'batch_inputs': inputs, 
        'batch_events': [filtered_events]
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[2]
    predictions = logits.argmax(dim=-1)[0].cpu().numpy().tolist()
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
    probabilities = [probabilities[idx][pred] for idx, pred in enumerate(predictions)]
    if len(new_events) > 1:
        assert len(predictions) == len(new_events) * (len(new_events) - 1) / 2
    return new_events, predictions, probabilities

def test(args, test_dataset, model, tokenizer, save_weights:list):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, batch_size=1, shuffle=False)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        td_metrics, ec_metrics = test_loop(args, test_dataloader, model)
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'at') as f:
            f.write(
                save_weight + '\n' + json.dumps(td_metrics, cls=NpEncoder) + 
                '\n\n' + json.dumps(ec_metrics, cls=NpEncoder) + '\n\n\n'
            )

if __name__ == '__main__':
    args = parse_args()
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            f'Output directory ({args.output_dir}) already exists and is not empty.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Prepare task
    args.id2label = {0:'O'}
    for c in SUBTYPES:
        args.id2label[len(args.id2label)] = f"B-{c}"
        args.id2label[len(args.id2label)] = f"I-{c}"
    args.label2id = {v: k for k, v in args.id2label.items()}
    args.trigger_num_labels = len(args.id2label)
    # Load pretrained model and tokenizer
    logger.info(f'using model {"with" if args.add_contrastive_loss else "without"} Contrastive loss')
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    config = AutoConfig.from_pretrained(args.model_checkpoint, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, cache_dir=args.cache_dir)
    args.num_labels = 2
    model = LongformerSoftmaxForEC.from_pretrained(
        args.model_checkpoint,
        config=config,
        cache_dir=args.cache_dir, 
        args=args
    ).to(args.device)
    # Training
    save_weights = []
    if args.do_train:
        train_dataset = KBPCoref(args.train_file)
        dev_dataset = KBPCoref(args.dev_file)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    if args.do_test:
        test_dataset = KBPCoref(args.test_file)
        test(args, test_dataset, model, tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        test_dataset = KBPCoref(args.test_file)
        for best_save_weight in save_weights:
            logger.info(f'loading weights from {best_save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, best_save_weight)))
            # predict triggers
            logger.info(f'predicting td labels of {best_save_weight}...')
            td_results = []
            model.eval()
            for sample in tqdm(test_dataset):
                pred_label = predict_td(args, sample['document'], model, tokenizer)
                td_results.append({
                    "doc_id": sample['id'], 
                    "document": sample['document'], 
                    "pred_label": pred_label, 
                    "true_label": sample['td_tags']
                })
            with open(os.path.join(args.output_dir, best_save_weight + '_test_pred_events.json'), 'wt', encoding='utf-8') as f:
                for example_result in td_results:
                    f.write(json.dumps(example_result) + '\n')
            # predict coreference
            logger.info(f'predicting coref labels of {best_save_weight}...')
            ec_results = []
            for sample in tqdm(td_results):
                events = [
                    [event['start'], event['start'] + len(event['trigger']) - 1] 
                    for event in sample['pred_label']
                ]
                new_events, predictions, probabilities = predict_ec(args, sample['document'], events, model, tokenizer)
                ec_results.append({
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
            with open(os.path.join(args.output_dir, best_save_weight + '_test_pred_corefs.json'), 'wt', encoding='utf-8') as f:
                for example_result in ec_results:
                    f.write(json.dumps(example_result) + '\n')
