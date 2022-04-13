import os
import sys
import logging
import torch
import json
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
import numpy as np
from sklearn.metrics import classification_report
sys.path.append('../../')
from src.tools import seed_everything, NpEncoder
from src.event_coref.arg import parse_args
from src.event_coref.data import KBPCoref, get_dataLoader
from src.event_coref.modeling import LongformerSoftmaxForEC

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")

def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(dataloader)
    
    model.train()
    no_tensor = ['batch_events', 'batch_event_cluster_ids']
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = {k: v if k in no_tensor else v.to(args.device) for k, v in batch_data.items()}
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
            no_tensor = ['batch_events', 'batch_event_cluster_ids']
            batch_data = {k: v if k in no_tensor else v.to(args.device) for k, v in batch_data.items()}
            _, pred, labels, masks = model(**batch_data)

            predictions = pred.argmax(dim=-1).cpu().numpy() # [batch, seq]
            y = labels.cpu().numpy()
            lens = np.sum(masks.cpu().numpy(), axis=-1)
            true_labels += [
                int(l) for label, seq_len in zip(y, lens) for idx, l in enumerate(label) if idx < seq_len
            ]
            true_predictions += [
                int(p) for pred, seq_len in zip(predictions, lens) for idx, p in enumerate(pred) if idx < seq_len
            ]
    return classification_report(true_labels, true_predictions, output_dict=True)

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
    best_f1 = 0.
    save_weights = []
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
            torch.save(
                model.state_dict(), 
                os.path.join(args.output_dir, save_weight)
            )
            save_weights.append(save_weight)
        with open(os.path.join(args.output_dir, 'dev_metrics.txt'), 'at') as f:
            f.write(f'epoch_{epoch+1}\n' + json.dumps(metrics, cls=NpEncoder) + '\n\n')
    logger.info("Done!")
    return save_weights

def predict(args, document:str, events:list, tokenizer, model):
    '''
    # Args:
        - events: [
            [e1_char_start, e1_char_end], ...
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
        token_end = inputs.char_to_token(char_end)
        if not token_end:
            continue
        filtered_events.append([token_start, token_end])
        new_events.append(event)
    inputs['batch_events'] = filtered_events.unsqueeze(0)
    no_tensor = ['batch_events', 'batch_event_cluster_ids']
    inputs = {k: v if k in no_tensor else v.to(args.device) for k, v in inputs.items()}
    with torch.no_grad():
        pred = model(**inputs)[1] # logits
    predictions = pred.argmax(dim=-1)[0].cpu().numpy().tolist()
    probabilities = torch.nn.functional.softmax(pred, dim=-1)[0].cpu().numpy().tolist()
    probabilities = [probabilities[idx][pred] for idx, pred in enumerate(predictions)]
    assert len(predictions) == len(new_events) * (len(new_events) - 1) / 2
    return new_events, predictions, probabilities

def test(args, test_dataset, tokenizer, model, save_weights:list=None):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, batch_size=1, shuffle=False)
    save_weights = save_weights if save_weights else args.save_weights
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'at') as f:
            f.write(f'{save_weight}\n{json.dumps(metrics, cls=NpEncoder)}\n\n')
        logger.info(f'predicting labels of {save_weight}...')
        results = []
        model.eval()
        for sample in tqdm(test_dataset):
            events = [[char_start, char_end] for _, char_start, char_end, _, _ in sample['events']]
            event_cluster_dic = {char_start:cluster_id for _, char_start, _, _, cluster_id in sample['events']}
            true_label = []
            for i in range(len(events) - 1):
                for j in range(i+1, len(events)):
                    true_label.append(1 if event_cluster_dic[events[i][0]] == event_cluster_dic[events[j][0]] else 0)
            new_events, predictions, probabilities = predict(args, sample['document'], events, tokenizer, model)
            results.append({
                "doc_id": sample['id'], 
                "document": sample['document'], 
                "events": [{
                    'start': char_start, 
                    'end': char_end, 
                    'trigger': sample['document'][char_start:char_end+1]
                } for char_start, char_end in new_events], 
                "pred_label": predictions, 
                "pred_prob": probabilities,
                "true_label": true_label
            })
        with open(os.path.join(args.output_dir, save_weight + '_test_pred.json'), 'wt', encoding='utf-8') as f:
            for exapmle_result in results:
                f.write(json.dumps(exapmle_result) + '\n')

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        args.model_checkpoint, 
        cache_dir=args.cache_dir, 
        num_labels=2
    )
    config.loss_type = args.softmax_loss
    config.use_device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, cache_dir=args.cache_dir)
    model = LongformerSoftmaxForEC.from_pretrained(
        args.model_checkpoint,
        config=config,
        cache_dir=args.cache_dir
    ).to(args.device)
    # Training
    if args.do_train:
        logger.info(f'Training/evaluation parameters: {args}')
        train_dataset = KBPCoref(args.train_file)
        dev_dataset = KBPCoref(args.dev_file)
        args.save_weights = train(args, train_dataset, dev_dataset, model, tokenizer)
    # Testing
    if args.do_test:
        test_dataset = KBPCoref(args.test_file)
        test(args, test_dataset, tokenizer, model)
