import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LongformerPreTrainedModel, LongformerModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from ..tools import LabelSmoothingCrossEntropy, FocalLoss

class LongformerSoftmaxForEC(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.coref_classifier = nn.Linear(3*config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.use_device = config.use_device
        self.post_init()
    
    def forward(self, input_ids, attention_mask, batch_events, batch_event_cluster_ids=None):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # construct event pairs (event_1, event_2)
        batch_event_1_list, batch_event_2_list, batch_event_mask = [], [], []
        max_len = 0
        if batch_event_cluster_ids:
            batch_coref_labels = []
            for events, event_cluster_ids in zip(batch_events, batch_event_cluster_ids):
                event_1_list, event_2_list, coref_labels = [], [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                max_len = max(max_len, len(coref_labels))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_event_mask.append([1] * len(coref_labels))
                batch_coref_labels.append(coref_labels)
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx])
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
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
                length = len(batch_event_mask[b_idx])
                batch_event_1_list[b_idx] += [[0, 0]] * (max_len - length)
                batch_event_2_list[b_idx] += [[0, 0]] * (max_len - length)
                batch_event_mask[b_idx] += [0] * (max_len - length)

        batch_event_1 = torch.tensor(batch_event_1_list).to(self.use_device)
        batch_event_2 = torch.tensor(batch_event_2_list).to(self.use_device)
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_labels = None
        if batch_event_cluster_ids:
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_event_1, span_indices_mask=batch_mask)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_event_2, span_indices_mask=batch_mask)
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
        logits = self.coref_classifier(batch_seq_reps)

        loss = None
        if batch_event_cluster_ids:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if batch_mask is not None:
                active_loss = batch_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = batch_labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), batch_labels.view(-1))
        return loss, logits, batch_labels, batch_mask
