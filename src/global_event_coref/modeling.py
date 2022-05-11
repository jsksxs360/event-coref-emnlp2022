import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import LongformerPreTrainedModel, LongformerModel
from transformers import BertModel, RobertaModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from ..tools import LabelSmoothingCrossEntropy, FocalLoss, SimpleTopicModel

MENTION_ENCODER = {
    'bert': BertModel, 
    'roberta': RobertaModel
}

class LongformerSoftmaxForEC(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.matching_style = args.matching_style
        if 'cosine' not in args.matching_style:
            if args.matching_style == 'base':
                multiples = 2
            elif args.matching_style in ['multi', 'dist']:
                multiples = 3
            elif args.matching_style == 'multi_dist':
                multiples = 4
            self.coref_classifier = nn.Linear(multiples * config.hidden_size, args.num_labels)
        else:
            self.cosine_space_dim = 64
            self.cosine_slices = 512
            self.tensor_factor = 2
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(config.hidden_size, self.cosine_space_dim)
            if args.matching_style == 'cosine':
                self.coref_classifier = nn.Linear(2 * config.hidden_size + self.cosine_slices, args.num_labels)
            elif args.matching_style == 'multi_cosine':
                self.coref_classifier = nn.Linear(3 * config.hidden_size + self.cosine_slices, args.num_labels)
            elif args.matching_style == 'multi_dist_cosine':
                self.coref_classifier = nn.Linear(4 * config.hidden_size + self.cosine_slices, args.num_labels)
        
        self.loss_type = args.softmax_loss
        self.add_contrastive_loss = args.add_contrastive_loss
        self.use_device = args.device
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
        elif self.matching_style == 'dist':
            batch_e1_e2_dist = torch.abs(batch_event_1_reps - batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_dist], dim=-1)
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_dist':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_dist = torch.abs(batch_event_1_reps - batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_e1_e2_dist], dim=-1)
        elif self.matching_style == 'multi_dist_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_dist = torch.abs(batch_event_1_reps - batch_event_2_reps)
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([
                batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_e1_e2_dist, batch_multi_cosine], 
                dim=-1
            )
        return batch_seq_reps

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
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
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
            loss_coref = loss_fct(active_logits, active_labels)
            if self.add_contrastive_loss:
                active_event_1_reps = batch_event_1_reps.view(-1, self.hidden_size)[active_loss]
                active_event_2_reps = batch_event_2_reps.view(-1, self.hidden_size)[active_loss]
                loss_contrasive = self._cal_circle_loss(active_event_1_reps, active_event_2_reps, active_labels)
                loss = 0.6 * loss_coref + 0.4 * loss_contrasive
            else:
                loss = loss_coref
        return loss, logits, batch_mask, batch_labels

class LongformerSoftmaxForECwithTopic(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.dist_dim = args.dist_dim
        self.hidden_size = config.hidden_size + args.topic_dim
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.topic_model = SimpleTopicModel(args=args)
        
        self.matching_style = args.matching_style
        if 'cosine' not in args.matching_style:
            if args.matching_style == 'base':
                multiples = 2
            elif args.matching_style in ['multi', 'dist']:
                multiples = 3
            elif args.matching_style == 'multi_dist':
                multiples = 4
            self.coref_classifier = nn.Linear(multiples * self.hidden_size, args.num_labels)
        else:
            self.cosine_space_dim = 64
            self.cosine_slices = 512
            self.tensor_factor = 2
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
            if args.matching_style == 'cosine':
                self.coref_classifier = nn.Linear(2 * self.hidden_size + self.cosine_slices, args.num_labels)
            elif args.matching_style == 'multi_cosine':
                self.coref_classifier = nn.Linear(3 * self.hidden_size + self.cosine_slices, args.num_labels)
            elif args.matching_style == 'multi_dist_cosine':
                self.coref_classifier = nn.Linear(4 * self.hidden_size + self.cosine_slices, args.num_labels)
        
        self.loss_type = args.softmax_loss
        self.add_contrastive_loss = args.add_contrastive_loss
        self.use_device = args.device
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
        elif self.matching_style == 'dist':
            batch_e1_e2_dist = torch.abs(batch_event_1_reps - batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_dist], dim=-1)
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_dist':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_dist = torch.abs(batch_event_1_reps - batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_e1_e2_dist], dim=-1)
        elif self.matching_style == 'multi_dist_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_dist = torch.abs(batch_event_1_reps - batch_event_2_reps)
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([
                batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_e1_e2_dist, batch_multi_cosine], 
                dim=-1
            )
        return batch_seq_reps
    
    def forward(self, batch_inputs, batch_events, batch_event_dists, batch_event_cluster_ids=None):
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # construct event pairs (event_1, event_2)
        batch_event_1_list, batch_event_2_list = [], []
        batch_event_1_dists, batch_event_2_dists = [], []
        batch_event_mask = []
        max_len = 0
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            for events, event_dists, event_cluster_ids in zip(
                batch_events, batch_event_dists, batch_event_cluster_ids
            ):
                event_1_list, event_2_list = [], []
                event_1_idx, event_2_idx = [], []
                coref_labels = []
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
                batch_event_1_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_event_2_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_coref_labels.append(coref_labels)
                batch_event_mask.append([1] * len(coref_labels))
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_event_1_dists[b_idx] = torch.cat([
                    batch_event_1_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_2_dists[b_idx] = torch.cat([
                    batch_event_2_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_coref_labels[b_idx] += [0] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        else:
            for events, event_dists in zip(batch_events, batch_event_dists):
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
                batch_event_1_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_event_2_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_mask.append([1] * len(event_1_list))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_event_1_dists[b_idx] = torch.cat([
                    batch_event_1_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_2_dists[b_idx] = torch.cat([
                    batch_event_2_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_mask[b_idx] += [0] * pad_length

        batch_event_1 = torch.tensor(batch_event_1_list).to(self.use_device)
        batch_event_2 = torch.tensor(batch_event_2_list).to(self.use_device)
        batch_event_1_dists = torch.cat(batch_event_1_dists, dim=0)
        batch_event_2_dists = torch.cat(batch_event_2_dists, dim=0)
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_labels = None
        if batch_event_cluster_ids is not None:
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_event_1, span_indices_mask=batch_mask)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_event_2, span_indices_mask=batch_mask)
        # generate event topics
        topic_loss, batch_e1_topics, batch_e2_topics = self.topic_model(batch_event_1_dists, batch_event_2_dists, batch_mask)
        # matching
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_e1_topics], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_e2_topics], dim=-1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
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
            loss_coref = loss_fct(active_logits, active_labels)
            if self.add_contrastive_loss:
                active_event_1_reps = batch_event_1_reps.view(-1, self.hidden_size)[active_loss]
                active_event_2_reps = batch_event_2_reps.view(-1, self.hidden_size)[active_loss]
                loss_contrasive = self._cal_circle_loss(active_event_1_reps, active_event_2_reps, active_labels)
                loss = 0.4 * loss_coref + 0.4 * topic_loss + 0.2 * loss_contrasive
            else:
                loss = 0.5 * loss_coref + 0.5 * topic_loss
        return loss, logits, batch_mask, batch_labels

class LongformerSoftmaxForECwithMask(LongformerPreTrainedModel):
    def __init__(self, config, encoder_config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.num_subtypes = args.num_subtypes
        self.hidden_size = config.hidden_size + encoder_config.hidden_size
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mention_encoder = MENTION_ENCODER[args.mention_encoder_type].from_pretrained(
            args.mention_encoder_checkpoint, 
            config=encoder_config, 
            add_pooling_layer=False, 
            cache_dir=args.cache_dir,
        )
        self.mention_dropout = nn.Dropout(encoder_config.hidden_dropout_prob)
        self.mention_span_extractor = SelfAttentiveSpanExtractor(input_dim=encoder_config.hidden_size)
        self.encoder_dim = encoder_config.hidden_size
        self.subtype_classifier = nn.Linear(encoder_config.hidden_size, args.num_subtypes)
        
        self.matching_style = args.matching_style
        if 'cosine' not in args.matching_style:
            if args.matching_style == 'base':
                multiples = 2
            elif args.matching_style in ['multi', 'dist']:
                multiples = 3
            elif args.matching_style == 'multi_dist':
                multiples = 4
            self.coref_classifier = nn.Linear(multiples * self.hidden_size, args.num_labels)
        else:
            self.cosine_space_dim = 64
            self.cosine_slices = 512
            self.tensor_factor = 2
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size, self.cosine_space_dim)
            if args.matching_style == 'cosine':
                self.coref_classifier = nn.Linear(2 * self.hidden_size + self.cosine_slices, args.num_labels)
            elif args.matching_style == 'multi_cosine':
                self.coref_classifier = nn.Linear(3 * self.hidden_size + self.cosine_slices, args.num_labels)
            elif args.matching_style == 'multi_dist_cosine':
                self.coref_classifier = nn.Linear(4 * self.hidden_size + self.cosine_slices, args.num_labels)
        
        self.loss_type = args.softmax_loss
        self.add_contrastive_loss = args.add_contrastive_loss
        self.use_device = args.device
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

    def forward(self, 
        batch_inputs, 
        batch_events, 
        batch_mention_inputs_with_mask, 
        batch_mention_events, 
        batch_event_cluster_ids=None, 
        batch_event_subtypes=None
        ):
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # get local event mask representations
        batch_local_event_mask_reps = []
        for mention_mask_inputs, mention_events in zip(batch_mention_inputs_with_mask, batch_mention_events):
            encoder_outputs = self.mention_encoder(**mention_mask_inputs)
            mention_mask_output = encoder_outputs[0]
            mention_mask_output = self.mention_dropout(mention_mask_output)
            mention_event_list = [[event] for event in mention_events]
            mention_event_list = torch.tensor(mention_event_list).to(self.use_device)
            local_event_mask_reps = self.mention_span_extractor(mention_mask_output, mention_event_list).squeeze(dim=1) # (event_num, dim)
            batch_local_event_mask_reps.append(local_event_mask_reps)

        # construct event pairs (event_1, event_2)
        batch_event_1_list, batch_event_2_list = [], []
        batch_local_event_1_mask_reps, batch_local_event_2_mask_reps = [], []
        batch_event_mask = []
        max_len = 0
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            batch_local_event_1_subtypes, batch_local_event_2_subtypes = [], []
            for events, local_event_mask_reps, event_cluster_ids, event_subtypes in zip(
                batch_events, batch_local_event_mask_reps, batch_event_cluster_ids, batch_event_subtypes
                ):
                event_1_list, event_2_list = [], []
                event_1_idx, event_2_idx = [], []
                coref_labels = []
                event_1_subtypes, event_2_subtypes = [], []
                for i in range(len(events) - 1):
                    for j in range(i + 1, len(events)):
                        event_1_list.append(events[i])
                        event_2_list.append(events[j])
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                        event_1_subtypes.append(event_subtypes[i])
                        event_2_subtypes.append(event_subtypes[j])
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                max_len = max(max_len, len(coref_labels))
                batch_event_1_list.append(event_1_list)
                batch_event_2_list.append(event_2_list)
                batch_local_event_1_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_local_event_1_subtypes.append(event_1_subtypes)
                batch_local_event_2_subtypes.append(event_2_subtypes)
                batch_coref_labels.append(coref_labels)
                batch_event_mask.append([1] * len(coref_labels))
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_local_event_1_mask_reps[b_idx] = torch.cat([
                    batch_local_event_1_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_mask_reps[b_idx] = torch.cat([
                    batch_local_event_2_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_1_subtypes[b_idx] += [0] * pad_length
                batch_local_event_2_subtypes[b_idx] += [0] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        else:
            for events, local_event_mask_reps in zip(batch_events, batch_local_event_mask_reps):
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
                batch_local_event_1_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_mask.append([1] * len(event_1_list))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_event_1_list[b_idx] += [[0, 0]] * pad_length
                batch_event_2_list[b_idx] += [[0, 0]] * pad_length
                batch_local_event_1_mask_reps[b_idx] = torch.cat([
                    batch_local_event_1_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_mask_reps[b_idx] = torch.cat([
                    batch_local_event_2_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_mask[b_idx] += [0] * pad_length

        batch_event_1 = torch.tensor(batch_event_1_list).to(self.use_device)
        batch_event_2 = torch.tensor(batch_event_2_list).to(self.use_device)
        batch_local_event_1_mask_reps = torch.cat(batch_local_event_1_mask_reps, dim=0)
        batch_local_event_2_mask_reps = torch.cat(batch_local_event_2_mask_reps, dim=0)
        event_1_subtypes_logits = self.subtype_classifier(batch_local_event_1_mask_reps)
        event_2_subtypes_logits = self.subtype_classifier(batch_local_event_2_mask_reps)
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_labels, batch_event_1_subtypes, batch_event_2_subtypes = None, None, None
        if batch_event_cluster_ids is not None:
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
            batch_event_1_subtypes = torch.tensor(batch_local_event_1_subtypes).to(self.use_device)
            batch_event_2_subtypes = torch.tensor(batch_local_event_2_subtypes).to(self.use_device)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_event_1, span_indices_mask=batch_mask)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_event_2, span_indices_mask=batch_mask)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_local_event_1_mask_reps], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_local_event_2_mask_reps], dim=-1)
        if self.matching_style == 'base':
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps], dim=-1)
        elif self.matching_style == 'multi':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi], dim=-1)
        elif self.matching_style == 'dist':
            batch_e1_e2_dist = torch.abs(batch_event_1_reps - batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_dist], dim=-1)
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_dist':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_dist = torch.abs(batch_event_1_reps - batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_e1_e2_dist], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_dist_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_e1_e2_dist = torch.abs(batch_event_1_reps - batch_event_2_reps)
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([
                batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_e1_e2_dist, batch_multi_cosine], 
                dim=-1
            )
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
            active_e_1_sutype_logits = event_1_subtypes_logits.view(-1, self.num_subtypes)[active_loss]
            active_e_2_sutype_logits = event_2_subtypes_logits.view(-1, self.num_subtypes)[active_loss]
            active_subtype_logits = torch.cat([active_e_1_sutype_logits, active_e_2_sutype_logits], dim=0)
            active_e_1_subtypes = batch_event_1_subtypes.view(-1)[active_loss]
            active_e_2_subtypes = batch_event_2_subtypes.view(-1)[active_loss]
            active_subtype_labels = torch.cat([active_e_1_subtypes, active_e_2_subtypes], dim=0)
            
            loss_subtype = loss_fct(active_subtype_logits, active_subtype_labels)
            loss_coref = loss_fct(active_logits, active_labels)
            if self.add_contrastive_loss:
                active_event_1_reps = batch_event_1_reps.view(-1, self.hidden_size)[active_loss]
                active_event_2_reps = batch_event_2_reps.view(-1, self.hidden_size)[active_loss]
                loss_contrasive = self._cal_circle_loss(active_event_1_reps, active_event_2_reps, active_labels)
                loss = 0.4 * loss_coref + 0.3 * loss_subtype + 0.3 * loss_contrasive
            else:
                loss = 0.5 * loss_coref + 0.5 * loss_subtype
        return loss, logits, batch_mask, batch_labels
