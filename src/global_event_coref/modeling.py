import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LongformerPreTrainedModel, LongformerModel
from transformers import BertModel, RobertaModel, DebertaModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from ..tools import LabelSmoothingCrossEntropy, FocalLoss

MENTION_ENCODER = {
    'bert': BertModel, 
    'roberta': RobertaModel
}

class LongformerSoftmaxForEC(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.coref_classifier = nn.Linear(3*config.hidden_size, args.num_labels)
        self.loss_type = args.loss_type
        self.use_device = args.use_device
        self.post_init()
    
    def forward(self, input_ids, attention_mask, batch_events, batch_event_cluster_ids=None):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # construct event pairs (event_1, event_2)
        batch_event_1_list, batch_event_2_list, batch_event_mask = [], [], []
        max_len = 0
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            for events, event_cluster_ids in zip(batch_events, batch_event_cluster_ids):
                event_1_list, event_2_list, coref_labels = [], [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                max_len = max(max_len, len(coref_labels))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_event_mask.append([1] * len(coref_labels))
                batch_coref_labels.append(coref_labels)
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
        else:
            for events in batch_events:
                event_1_list, event_2_list = [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                max_len = max(max_len, len(event_1_list))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_event_mask.append([1] * len(event_1_list))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length

        batch_event_1 = torch.tensor(batch_event_1_list).to(self.use_device)
        batch_event_2 = torch.tensor(batch_event_2_list).to(self.use_device)
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_labels = None
        if batch_event_cluster_ids is not None:
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_event_1, span_indices_mask=batch_mask)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_event_2, span_indices_mask=batch_mask)
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
        logits = self.coref_classifier(batch_seq_reps)

        loss = None
        if batch_event_cluster_ids is not None and max_len > 0:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = batch_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = batch_labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        return loss, logits, batch_mask, batch_labels

class LongformerSoftmaxContrastiveForEC(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.coref_classifier = nn.Linear(3*config.hidden_size, args.num_labels)
        self.loss_type = args.loss_type
        self.use_device = args.use_device
        self.post_init()
    
    def _cal_circle_loss(self, event_1_reps, event_2_reps, coref_labels, l=20.):
        norms_1 = (event_1_reps ** 2).sum(axis=1, keepdims=True) ** 0.5
        event_1_reps = event_1_reps / norms_1
        norms_2 = (event_2_reps ** 2).sum(axis=1, keepdims=True) ** 0.5
        event_2_reps = event_2_reps / norms_2
        event_cos = torch.sum(event_1_reps * event_2_reps, dim=1) * l
        # calculate the difference between each pair of Cosine values
        event_cos_diff = event_cos[:, None] - event_cos[None, :]
        # find noncoref-coref index
        select_idx = coref_labels[:, None] < coref_labels[None, :]
        select_idx = select_idx.float()

        event_cos_diff = event_cos_diff - (1 - select_idx) * 1e12
        event_cos_diff = event_cos_diff.view(-1)
        event_cos_diff = torch.cat((torch.tensor([0.0], device=self.use_device), event_cos_diff), dim=0)
        return torch.logsumexp(event_cos_diff, dim=0)

    def forward(self, input_ids, attention_mask, batch_events, batch_event_cluster_ids=None):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # construct event pairs (event_1, event_2)
        batch_event_1_list, batch_event_2_list, batch_event_mask = [], [], []
        max_len = 0
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            for events, event_cluster_ids in zip(batch_events, batch_event_cluster_ids):
                event_1_list, event_2_list, coref_labels = [], [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                max_len = max(max_len, len(coref_labels))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_event_mask.append([1] * len(coref_labels))
                batch_coref_labels.append(coref_labels)
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
        else:
            for events in batch_events:
                event_1_list, event_2_list = [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                max_len = max(max_len, len(event_1_list))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_event_mask.append([1] * len(event_1_list))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length

        batch_event_1 = torch.tensor(batch_event_1_list).to(self.use_device)
        batch_event_2 = torch.tensor(batch_event_2_list).to(self.use_device)
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_labels = None
        if batch_event_cluster_ids is not None:
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_event_1, span_indices_mask=batch_mask)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_event_2, span_indices_mask=batch_mask)
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
        logits = self.coref_classifier(batch_seq_reps)

        loss = None
        if batch_event_cluster_ids is not None and max_len > 0:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = batch_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = batch_labels.view(-1)[active_loss]
            active_event_1_reps = batch_event_1_reps.view(-1, self.hidden_size)[active_loss]
            active_event_2_reps = batch_event_2_reps.view(-1, self.hidden_size)[active_loss]
            loss_coref = loss_fct(active_logits, active_labels)
            loss_contrasive = self._cal_circle_loss(active_event_1_reps, active_event_2_reps, active_labels)
            loss = 0.5 * loss_coref + 0.5 * loss_contrasive
        return loss, logits, batch_mask, batch_labels

class LongformerSoftmaxForECwithMention(LongformerPreTrainedModel):
    def __init__(self, config, encoder_config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        
        if args.mention_encoder_type == 'deberta':
            self.mention_encoder = DebertaModel.from_pretrained(
                args.mention_encoder_checkpoint, 
                config=encoder_config, 
                cache_dir=args.cache_dir,
            )
        else:
            self.mention_encoder = MENTION_ENCODER[args.mention_encoder_type].from_pretrained(
                args.mention_encoder_checkpoint, 
                config=encoder_config, 
                add_pooling_layer=False, 
                cache_dir=args.cache_dir,
            )
        self.mention_dropout = nn.Dropout(encoder_config.hidden_dropout_prob)
        self.mention_span_extractor = SelfAttentiveSpanExtractor(input_dim=encoder_config.hidden_size)
        self.encoder_dim = encoder_config.hidden_size

        self.coref_classifier = nn.Linear(3 * (config.hidden_size + encoder_config.hidden_size), args.num_labels)
        self.loss_type = args.loss_type
        self.use_device = args.use_device
        self.post_init()
    
    def forward(self, batch_inputs, batch_events, batch_mention_inputs, batch_mention_events, batch_event_cluster_ids=None):
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # get local event representations
        batch_local_event_reps = []
        for mention_inputs, mention_events in zip(batch_mention_inputs, batch_mention_events):
            encoder_outputs = self.mention_encoder(**mention_inputs)
            mention_output = encoder_outputs[0]
            mention_output = self.mention_dropout(mention_output)
            mention_event_list = [[event] for event in mention_events]
            mention_event_list = torch.tensor(mention_event_list).to(self.use_device)
            local_event_reps = self.mention_span_extractor(mention_output, mention_event_list).squeeze(dim=1) # (event_num, dim)
            batch_local_event_reps.append(local_event_reps)

        # construct event pairs (event_1, event_2)
        batch_event_1_list, batch_event_2_list = [], []
        batch_local_event_1_reps, batch_local_event_2_reps = [], []
        batch_event_mask = []
        max_len = 0
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            for events, local_event_reps, event_cluster_ids in zip(
                batch_events, batch_local_event_reps, batch_event_cluster_ids
                ):
                event_1_list, event_2_list, coref_labels = [], [], []
                event_1_idx, event_2_idx = [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                max_len = max(max_len, len(coref_labels))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_local_event_1_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_mask.append([1] * len(coref_labels))
                batch_coref_labels.append(coref_labels)
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_local_event_1_reps[b_idx] = torch.cat([
                    batch_local_event_1_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_reps[b_idx] = torch.cat([
                    batch_local_event_2_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_mask[b_idx] += [0] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
        else:
            for events, local_event_reps in zip(batch_events, batch_local_event_reps):
                event_1_list, event_2_list = [], []
                event_1_idx, event_2_idx = [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                max_len = max(max_len, len(event_1_list))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_local_event_1_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_mask.append([1] * len(event_1_list))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_local_event_1_reps[b_idx] = torch.cat([
                    batch_local_event_1_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_reps[b_idx] = torch.cat([
                    batch_local_event_2_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_mask[b_idx] += [0] * pad_length

        batch_event_1 = torch.tensor(batch_event_1_list).to(self.use_device)
        batch_event_2 = torch.tensor(batch_event_2_list).to(self.use_device)
        batch_local_event_1_reps = torch.cat(batch_local_event_1_reps, dim=0)
        batch_local_event_2_reps = torch.cat(batch_local_event_2_reps, dim=0)
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_labels = None
        if batch_event_cluster_ids is not None:
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_event_1, span_indices_mask=batch_mask)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_event_2, span_indices_mask=batch_mask)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_local_event_1_reps], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_local_event_2_reps], dim=-1)
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
        logits = self.coref_classifier(batch_seq_reps)

        loss = None
        if batch_event_cluster_ids is not None and max_len > 0:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = batch_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = batch_labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
        return loss, logits, batch_mask, batch_labels
