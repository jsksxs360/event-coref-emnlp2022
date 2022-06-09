import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LongformerPreTrainedModel, LongformerModel
from transformers import BertModel, RobertaModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from ..tools import LabelSmoothingCrossEntropy, FocalLoss
from ..tools import SimpleTopicModel, SimpleTopicModelwithBN, SimpleTopicVMFModel

MENTION_ENCODER = {
    'bert': BertModel, 
    'roberta': RobertaModel
}
TOPIC_MODEL = {
    'stm': SimpleTopicModel, 
    'stm_bn': SimpleTopicModelwithBN, 
    'vmf': SimpleTopicVMFModel
}
COSINE_SPACE_DIM = 64
COSINE_SLICES = 128
COSINE_FACTOR = 4

class LongformerSoftmaxForEC(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.trigger_num_labels = args.trigger_num_labels
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.loss_type = args.softmax_loss
        self.add_contrastive_loss = args.add_contrastive_loss
        self.use_device = args.device
        # encoder & pooler
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.td_classifier = nn.Linear(self.hidden_size, self.trigger_num_labels)
        # event matching
        self.matching_style = args.matching_style
        if 'cosine' not in self.matching_style:
            if self.matching_style == 'base':
                multiples = 2
            elif self.matching_style == 'multi':
                multiples = 3
            self.coref_classifier = nn.Linear(multiples * self.hidden_size, self.num_labels)
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
            if self.matching_style == 'cosine':
                self.coref_classifier = nn.Linear(2 * self.hidden_size + self.cosine_slices, self.num_labels)
            elif self.matching_style == 'multi_cosine':
                self.coref_classifier = nn.Linear(3 * self.hidden_size + self.cosine_slices, self.num_labels)
            elif self.matching_style == 'multi_dist_cosine':
                self.coref_classifier = nn.Linear(4 * self.hidden_size + self.cosine_slices, self.num_labels)
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=2)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 1, 3, 2))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 1, 3, 2))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=2)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 1, 3, 2))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 1, 3, 2))
        # vector normalization
        norms_2 = (batch_event_2_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_2_reps = batch_event_2_reps / norms_2

        return torch.sum(batch_event_1_reps * batch_event_2_reps, dim=-1)
    
    def _cal_circle_loss(self, event_1_reps, event_2_reps, coref_labels, l=20.):
        norms_1 = (event_1_reps ** 2).sum(axis=1, keepdims=True) ** 0.5
        event_1_reps = event_1_reps / norms_1
        norms_2 = (event_2_reps ** 2).sum(axis=1, keepdims=True) ** 0.5
        event_2_reps = event_2_reps / norms_2
        event_cos = torch.sum(event_1_reps * event_2_reps, dim=1) * l
        # calculate the difference between each pair of Cosine values
        event_cos_diff = event_cos[:, None] - event_cos[None, :]
        # find (noncoref, coref) index
        select_idx = coref_labels[:, None] < coref_labels[None, :]
        select_idx = select_idx.float()

        event_cos_diff = event_cos_diff - (1 - select_idx) * 1e12
        event_cos_diff = event_cos_diff.view(-1)
        event_cos_diff = torch.cat((torch.tensor([0.0], device=self.use_device), event_cos_diff), dim=0)
        return torch.logsumexp(event_cos_diff, dim=0)

    def _matching_func(self, batch_event_1_reps, batch_event_2_reps):
        if self.matching_style == 'base':
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps], dim=-1)
        elif self.matching_style == 'multi':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_dist_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_dist = torch.abs(batch_event_1_reps - batch_event_2_reps)
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_e1_e2_dist, batch_multi_cosine], dim=-1)
        return batch_seq_reps

    def forward(self, batch_inputs, batch_events=None, batch_td_labels=None, batch_event_cluster_ids=None):
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # predict trigger
        td_logits = self.td_classifier(sequence_output)
        if batch_events is None:
            return None, td_logits
        # construct event pairs (event_1, event_2)
        batch_event_1_list, batch_event_2_list = [], []
        max_len, batch_event_mask = 0, []
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
                batch_coref_labels.append(coref_labels)
                batch_event_mask.append([1] * len(coref_labels))
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
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
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        # extract events & predict coref
        batch_event_1 = torch.tensor(batch_event_1_list).to(self.use_device)
        batch_event_2 = torch.tensor(batch_event_2_list).to(self.use_device)
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_event_1_reps = self.span_extractor(sequence_output, batch_event_1, span_indices_mask=batch_mask)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_event_2, span_indices_mask=batch_mask)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        coref_logits = self.coref_classifier(batch_seq_reps)
        # calculate loss 
        loss, batch_ec_labels = None, None
        attention_mask = batch_inputs['attention_mask']
        if batch_event_cluster_ids is not None and max_len > 0:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            # trigger detection loss
            active_td_loss = attention_mask.view(-1) == 1
            active_td_logits = td_logits.view(-1, self.trigger_num_labels)[active_td_loss]
            active_td_labels = batch_td_labels.view(-1)[active_td_loss]
            loss_td = loss_fct(active_td_logits, active_td_labels)
            # event coreference loss
            active_coref_loss = batch_mask.view(-1) == 1
            active_coref_logits = coref_logits.view(-1, self.num_labels)[active_coref_loss]
            batch_ec_labels = torch.tensor(batch_coref_labels).to(self.use_device)
            active_coref_labels = batch_ec_labels.view(-1)[active_coref_loss]
            loss_coref = loss_fct(active_coref_logits, active_coref_labels)
            if self.add_contrastive_loss:
                active_event_1_reps = batch_event_1_reps.view(-1, self.hidden_size)[active_coref_loss]
                active_event_2_reps = batch_event_2_reps.view(-1, self.hidden_size)[active_coref_loss]
                loss_contrasive = self._cal_circle_loss(active_event_1_reps, active_event_2_reps, active_coref_labels)
                loss = torch.log(1 + loss_td) + torch.log(1 + loss_coref) + 0.2 * loss_contrasive
            else:
                loss = torch.log(1 + loss_td) + torch.log(1 + loss_coref)
        return loss, td_logits, coref_logits, attention_mask, batch_td_labels, batch_mask, batch_ec_labels
