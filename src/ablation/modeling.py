import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, RobertaPreTrainedModel
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

class WithoutGlobalEncoder(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.dist_dim = args.dist_dim
        self.num_subtypes = args.num_subtypes
        self.hidden_size = config.hidden_size + args.topic_dim
        self.mention_encoder_dim = config.hidden_size
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        # encoder & pooler
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mention_span_extractor = SelfAttentiveSpanExtractor(input_dim=self.mention_encoder_dim)
        self.subtype_classifier = nn.Linear(self.mention_encoder_dim, self.num_subtypes)
        self.topic_model = TOPIC_MODEL[args.topic_model](args=args)
        # event matching
        self.matching_style = args.matching_style
        if 'cosine' not in self.matching_style:
            if self.matching_style == 'base':
                multiples = 2
            elif self.matching_style == 'multi' or self.matching_style == 'dist':
                multiples = 3
            elif self.matching_style == 'multi_dist':
                multiples = 4
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
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_e1_e2_dist, batch_multi_cosine], dim=-1)
        return batch_seq_reps

    def forward(self, 
        batch_mention_inputs_with_mask, 
        batch_mention_events, 
        batch_event_dists, 
        batch_event_cluster_ids=None, 
        batch_event_subtypes=None
        ):
        # construct local event mask representations
        batch_local_event_mask_reps = []
        for mention_mask_inputs, mention_events in zip(batch_mention_inputs_with_mask, batch_mention_events):
            encoder_outputs = self.bert(**mention_mask_inputs)
            mention_mask_output = encoder_outputs[0]
            mention_mask_output = self.dropout(mention_mask_output)
            mention_event_list = [[event] for event in mention_events]
            mention_event_list = torch.tensor(mention_event_list).to(self.use_device)
            local_event_mask_reps = self.mention_span_extractor(mention_mask_output, mention_event_list).squeeze(dim=1) # (event_num, dim)
            batch_local_event_mask_reps.append(local_event_mask_reps)
        # construct event pairs (event_1, event_2)
        batch_local_event_1_mask_reps, batch_local_event_2_mask_reps = [], []
        batch_event_1_dists, batch_event_2_dists = [], []
        max_len, batch_event_mask = 0, []
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            batch_local_event_1_subtypes, batch_local_event_2_subtypes = [], []
            for local_event_mask_reps, event_dists, event_cluster_ids, event_subtypes in zip(
                batch_local_event_mask_reps, batch_event_dists, batch_event_cluster_ids, batch_event_subtypes
            ):
                event_num = local_event_mask_reps.size()[0]
                event_1_idx, event_2_idx = [], []
                coref_labels = []
                event_1_subtypes, event_2_subtypes = [], []
                for i in range(event_num - 1):
                    for j in range(i + 1, event_num):
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                        event_1_subtypes.append(event_subtypes[i])
                        event_2_subtypes.append(event_subtypes[j])
                max_len = max(max_len, len(coref_labels))
                batch_local_event_1_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_1_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_event_2_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_local_event_1_subtypes.append(event_1_subtypes)
                batch_local_event_2_subtypes.append(event_2_subtypes)
                batch_coref_labels.append(coref_labels)
                batch_event_mask.append([1] * len(coref_labels))
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_local_event_1_mask_reps[b_idx] = torch.cat([
                    batch_local_event_1_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_mask_reps[b_idx] = torch.cat([
                    batch_local_event_2_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_1_dists[b_idx] = torch.cat([
                    batch_event_1_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_2_dists[b_idx] = torch.cat([
                    batch_event_2_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_1_subtypes[b_idx] += [0] * pad_length
                batch_local_event_2_subtypes[b_idx] += [0] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        else:
            for local_event_mask_reps, event_dists in zip(
                batch_local_event_mask_reps, batch_event_dists
            ):
                event_num = local_event_mask_reps.size()[0]
                event_1_idx, event_2_idx = [], []
                for i in range(event_num - 1):
                    for j in range(i + 1, event_num):
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                max_len = max(max_len, len(event_1_idx))
                batch_local_event_1_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_1_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_event_2_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_mask.append([1] * len(event_1_idx))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_local_event_1_mask_reps[b_idx] = torch.cat([
                    batch_local_event_1_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_mask_reps[b_idx] = torch.cat([
                    batch_local_event_2_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_1_dists[b_idx] = torch.cat([
                    batch_event_1_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_2_dists[b_idx] = torch.cat([
                    batch_event_2_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_mask[b_idx] += [0] * pad_length
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        # predict event subtype
        batch_local_event_1_mask_reps = torch.cat(batch_local_event_1_mask_reps, dim=0)
        batch_local_event_2_mask_reps = torch.cat(batch_local_event_2_mask_reps, dim=0)
        event_1_subtypes_logits = self.subtype_classifier(batch_local_event_1_mask_reps)
        event_2_subtypes_logits = self.subtype_classifier(batch_local_event_2_mask_reps)
        # generate event topics
        batch_event_1_dists = torch.cat(batch_event_1_dists, dim=0)
        batch_event_2_dists = torch.cat(batch_event_2_dists, dim=0)
        loss_topic, batch_e1_topics, batch_e2_topics = self.topic_model(batch_event_1_dists, batch_event_2_dists, batch_mask)
        # matching & predict coref
        batch_event_1_reps = torch.cat([batch_local_event_1_mask_reps, batch_e1_topics], dim=-1)
        batch_event_2_reps = torch.cat([batch_local_event_2_mask_reps, batch_e2_topics], dim=-1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        # calculate loss 
        loss, batch_labels = None, None
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
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
            active_labels = batch_labels.view(-1)[active_loss]
            active_e_1_sutype_logits = event_1_subtypes_logits.view(-1, self.num_subtypes)[active_loss]
            active_e_2_sutype_logits = event_2_subtypes_logits.view(-1, self.num_subtypes)[active_loss]
            active_subtype_logits = torch.cat([active_e_1_sutype_logits, active_e_2_sutype_logits], dim=0)
            batch_event_1_subtypes = torch.tensor(batch_local_event_1_subtypes).to(self.use_device)
            batch_event_2_subtypes = torch.tensor(batch_local_event_2_subtypes).to(self.use_device)
            active_e_1_subtypes = batch_event_1_subtypes.view(-1)[active_loss]
            active_e_2_subtypes = batch_event_2_subtypes.view(-1)[active_loss]
            active_subtype_labels = torch.cat([active_e_1_subtypes, active_e_2_subtypes], dim=0)
            
            loss_subtype = loss_fct(active_subtype_logits, active_subtype_labels)
            loss_coref = loss_fct(active_logits, active_labels)
            loss = torch.log(1 + loss_coref) + torch.log(1 + loss_subtype) + torch.log(1 + loss_topic)
        return loss, logits, batch_mask, batch_labels

class ChunkBertEncoder(BertPreTrainedModel):
    def __init__(self, config, encoder_config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.dist_dim = args.dist_dim
        self.num_subtypes = args.num_subtypes
        self.hidden_size = config.hidden_size + encoder_config.hidden_size + args.topic_dim
        self.encoder_dim = config.hidden_size
        self.mention_encoder_dim = encoder_config.hidden_size
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        # encoder & pooler
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mention_encoder = MENTION_ENCODER[args.mention_encoder_type].from_pretrained(
            args.mention_encoder_checkpoint, 
            config=encoder_config, 
            add_pooling_layer=False, 
            cache_dir=args.cache_dir,
        )
        self.mention_dropout = nn.Dropout(encoder_config.hidden_dropout_prob)
        self.mention_span_extractor = SelfAttentiveSpanExtractor(input_dim=self.mention_encoder_dim)
        self.subtype_classifier = nn.Linear(self.mention_encoder_dim, self.num_subtypes)
        self.topic_model = TOPIC_MODEL[args.topic_model](args=args)
        # event matching
        self.matching_style = args.matching_style
        if 'cosine' not in self.matching_style:
            if self.matching_style == 'base':
                multiples = 2
            elif self.matching_style == 'multi' or self.matching_style == 'dist':
                multiples = 3
            elif self.matching_style == 'multi_dist':
                multiples = 4
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
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_e1_e2_dist, batch_multi_cosine], dim=-1)
        return batch_seq_reps

    def forward(self, 
        batch_mention_inputs, 
        batch_mention_events, 
        batch_mention_inputs_with_mask, 
        batch_mention_events_with_mask, 
        batch_event_dists, 
        batch_event_cluster_ids=None, 
        batch_event_subtypes=None
        ):
        # construct local event mask representations
        batch_local_event_reps, batch_local_event_mask_reps = [], []
        for mention_inputs, mention_events, mention_mask_inputs, mention_mask_events in zip(
            batch_mention_inputs, batch_mention_events, batch_mention_inputs_with_mask, batch_mention_events_with_mask
        ):
            outputs = self.bert(**mention_inputs)
            mention_output = outputs[0]
            mention_output = self.dropout(mention_output)
            mention_event_list = [[event] for event in mention_events]
            mention_event_list = torch.tensor(mention_event_list).to(self.use_device)
            local_event_reps = self.span_extractor(mention_output, mention_event_list).squeeze(dim=1) # (event_num, dim)
            encoder_outputs = self.mention_encoder(**mention_mask_inputs)
            mention_mask_output = encoder_outputs[0]
            mention_mask_output = self.mention_dropout(mention_mask_output)
            mention_mask_event_list = [[event] for event in mention_mask_events]
            mention_mask_event_list = torch.tensor(mention_mask_event_list).to(self.use_device)
            local_event_mask_reps = self.mention_span_extractor(mention_mask_output, mention_mask_event_list).squeeze(dim=1) # (event_num, dim)
            batch_local_event_reps.append(local_event_reps)
            batch_local_event_mask_reps.append(local_event_mask_reps)
        # construct event pairs (event_1, event_2)
        batch_local_event_1_reps, batch_local_event_2_reps = [], []
        batch_local_event_1_mask_reps, batch_local_event_2_mask_reps = [], []
        batch_event_1_dists, batch_event_2_dists = [], []
        max_len, batch_event_mask = 0, []
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            batch_local_event_1_subtypes, batch_local_event_2_subtypes = [], []
            for local_event_reps, local_event_mask_reps, event_dists, event_cluster_ids, event_subtypes in zip(
                batch_local_event_reps, batch_local_event_mask_reps, batch_event_dists, batch_event_cluster_ids, batch_event_subtypes
            ):
                event_num = local_event_mask_reps.size()[0]
                event_1_idx, event_2_idx = [], []
                coref_labels = []
                event_1_subtypes, event_2_subtypes = [], []
                for i in range(event_num - 1):
                    for j in range(i + 1, event_num):
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                        event_1_subtypes.append(event_subtypes[i])
                        event_2_subtypes.append(event_subtypes[j])
                max_len = max(max_len, len(coref_labels))
                batch_local_event_1_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_local_event_1_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_1_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_event_2_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_local_event_1_subtypes.append(event_1_subtypes)
                batch_local_event_2_subtypes.append(event_2_subtypes)
                batch_coref_labels.append(coref_labels)
                batch_event_mask.append([1] * len(coref_labels))
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_local_event_1_reps[b_idx] = torch.cat([
                    batch_local_event_1_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_reps[b_idx] = torch.cat([
                    batch_local_event_2_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_1_mask_reps[b_idx] = torch.cat([
                    batch_local_event_1_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_mask_reps[b_idx] = torch.cat([
                    batch_local_event_2_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_1_dists[b_idx] = torch.cat([
                    batch_event_1_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_2_dists[b_idx] = torch.cat([
                    batch_event_2_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_1_subtypes[b_idx] += [0] * pad_length
                batch_local_event_2_subtypes[b_idx] += [0] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        else:
            for local_event_reps, local_event_mask_reps, event_dists in zip(
                batch_local_event_reps, batch_local_event_mask_reps, batch_event_dists
            ):
                event_num = local_event_mask_reps.size()[0]
                event_1_idx, event_2_idx = [], []
                for i in range(event_num - 1):
                    for j in range(i + 1, event_num):
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                max_len = max(max_len, len(event_1_idx))
                batch_local_event_1_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_local_event_1_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_1_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_event_2_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_mask.append([1] * len(event_1_idx))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_local_event_1_reps[b_idx] = torch.cat([
                    batch_local_event_1_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_reps[b_idx] = torch.cat([
                    batch_local_event_2_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_1_mask_reps[b_idx] = torch.cat([
                    batch_local_event_1_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_mask_reps[b_idx] = torch.cat([
                    batch_local_event_2_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_1_dists[b_idx] = torch.cat([
                    batch_event_1_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_2_dists[b_idx] = torch.cat([
                    batch_event_2_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_mask[b_idx] += [0] * pad_length
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_local_event_1_reps = torch.cat(batch_local_event_1_reps, dim=0)
        batch_local_event_2_reps = torch.cat(batch_local_event_2_reps, dim=0)
        # predict event subtype
        batch_local_event_1_mask_reps = torch.cat(batch_local_event_1_mask_reps, dim=0)
        batch_local_event_2_mask_reps = torch.cat(batch_local_event_2_mask_reps, dim=0)
        event_1_subtypes_logits = self.subtype_classifier(batch_local_event_1_mask_reps)
        event_2_subtypes_logits = self.subtype_classifier(batch_local_event_2_mask_reps)
        # generate event topics
        batch_event_1_dists = torch.cat(batch_event_1_dists, dim=0)
        batch_event_2_dists = torch.cat(batch_event_2_dists, dim=0)
        loss_topic, batch_e1_topics, batch_e2_topics = self.topic_model(batch_event_1_dists, batch_event_2_dists, batch_mask)
        # matching & predict coref
        batch_event_1_reps = torch.cat([batch_local_event_1_reps, batch_local_event_1_mask_reps, batch_e1_topics], dim=-1)
        batch_event_2_reps = torch.cat([batch_local_event_2_reps, batch_local_event_2_mask_reps, batch_e2_topics], dim=-1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        # calculate loss 
        loss, batch_labels = None, None
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
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
            active_labels = batch_labels.view(-1)[active_loss]
            active_e_1_sutype_logits = event_1_subtypes_logits.view(-1, self.num_subtypes)[active_loss]
            active_e_2_sutype_logits = event_2_subtypes_logits.view(-1, self.num_subtypes)[active_loss]
            active_subtype_logits = torch.cat([active_e_1_sutype_logits, active_e_2_sutype_logits], dim=0)
            batch_event_1_subtypes = torch.tensor(batch_local_event_1_subtypes).to(self.use_device)
            batch_event_2_subtypes = torch.tensor(batch_local_event_2_subtypes).to(self.use_device)
            active_e_1_subtypes = batch_event_1_subtypes.view(-1)[active_loss]
            active_e_2_subtypes = batch_event_2_subtypes.view(-1)[active_loss]
            active_subtype_labels = torch.cat([active_e_1_subtypes, active_e_2_subtypes], dim=0)
            
            loss_subtype = loss_fct(active_subtype_logits, active_subtype_labels)
            loss_coref = loss_fct(active_logits, active_labels)
            loss = torch.log(1 + loss_coref) + torch.log(1 + loss_subtype) + torch.log(1 + loss_topic)
        return loss, logits, batch_mask, batch_labels

class ChunkRobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config, encoder_config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.dist_dim = args.dist_dim
        self.num_subtypes = args.num_subtypes
        self.hidden_size = config.hidden_size + encoder_config.hidden_size + args.topic_dim
        self.encoder_dim = config.hidden_size
        self.mention_encoder_dim = encoder_config.hidden_size
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        # encoder & pooler
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mention_encoder = MENTION_ENCODER[args.mention_encoder_type].from_pretrained(
            args.mention_encoder_checkpoint, 
            config=encoder_config, 
            add_pooling_layer=False, 
            cache_dir=args.cache_dir,
        )
        self.mention_dropout = nn.Dropout(encoder_config.hidden_dropout_prob)
        self.mention_span_extractor = SelfAttentiveSpanExtractor(input_dim=self.mention_encoder_dim)
        self.subtype_classifier = nn.Linear(self.mention_encoder_dim, self.num_subtypes)
        self.topic_model = TOPIC_MODEL[args.topic_model](args=args)
        # event matching
        self.matching_style = args.matching_style
        if 'cosine' not in self.matching_style:
            if self.matching_style == 'base':
                multiples = 2
            elif self.matching_style == 'multi' or self.matching_style == 'dist':
                multiples = 3
            elif self.matching_style == 'multi_dist':
                multiples = 4
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
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_e1_e2_dist, batch_multi_cosine], dim=-1)
        return batch_seq_reps

    def forward(self, 
        batch_mention_inputs, 
        batch_mention_events, 
        batch_mention_inputs_with_mask, 
        batch_mention_events_with_mask, 
        batch_event_dists, 
        batch_event_cluster_ids=None, 
        batch_event_subtypes=None
        ):
        # construct local event mask representations
        batch_local_event_reps, batch_local_event_mask_reps = [], []
        for mention_inputs, mention_events, mention_mask_inputs, mention_mask_events in zip(
            batch_mention_inputs, batch_mention_events, batch_mention_inputs_with_mask, batch_mention_events_with_mask
        ):
            outputs = self.roberta(**mention_inputs)
            mention_output = outputs[0]
            mention_output = self.dropout(mention_output)
            mention_event_list = [[event] for event in mention_events]
            mention_event_list = torch.tensor(mention_event_list).to(self.use_device)
            local_event_reps = self.span_extractor(mention_output, mention_event_list).squeeze(dim=1) # (event_num, dim)
            encoder_outputs = self.mention_encoder(**mention_mask_inputs)
            mention_mask_output = encoder_outputs[0]
            mention_mask_output = self.mention_dropout(mention_mask_output)
            mention_mask_event_list = [[event] for event in mention_mask_events]
            mention_mask_event_list = torch.tensor(mention_mask_event_list).to(self.use_device)
            local_event_mask_reps = self.mention_span_extractor(mention_mask_output, mention_mask_event_list).squeeze(dim=1) # (event_num, dim)
            batch_local_event_reps.append(local_event_reps)
            batch_local_event_mask_reps.append(local_event_mask_reps)
        # construct event pairs (event_1, event_2)
        batch_local_event_1_reps, batch_local_event_2_reps = [], []
        batch_local_event_1_mask_reps, batch_local_event_2_mask_reps = [], []
        batch_event_1_dists, batch_event_2_dists = [], []
        max_len, batch_event_mask = 0, []
        if batch_event_cluster_ids is not None:
            batch_coref_labels = []
            batch_local_event_1_subtypes, batch_local_event_2_subtypes = [], []
            for local_event_reps, local_event_mask_reps, event_dists, event_cluster_ids, event_subtypes in zip(
                batch_local_event_reps, batch_local_event_mask_reps, batch_event_dists, batch_event_cluster_ids, batch_event_subtypes
            ):
                event_num = local_event_mask_reps.size()[0]
                event_1_idx, event_2_idx = [], []
                coref_labels = []
                event_1_subtypes, event_2_subtypes = [], []
                for i in range(event_num - 1):
                    for j in range(i + 1, event_num):
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                        cluster_id_1, cluster_id_2 = event_cluster_ids[i], event_cluster_ids[j]
                        coref_labels.append(1 if cluster_id_1 == cluster_id_2 else 0)
                        event_1_subtypes.append(event_subtypes[i])
                        event_2_subtypes.append(event_subtypes[j])
                max_len = max(max_len, len(coref_labels))
                batch_local_event_1_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_local_event_1_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_1_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_event_2_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_local_event_1_subtypes.append(event_1_subtypes)
                batch_local_event_2_subtypes.append(event_2_subtypes)
                batch_coref_labels.append(coref_labels)
                batch_event_mask.append([1] * len(coref_labels))
            # padding
            for b_idx in range(len(batch_coref_labels)):
                pad_length = max_len - len(batch_coref_labels[b_idx]) if max_len > 0 else 1
                batch_local_event_1_reps[b_idx] = torch.cat([
                    batch_local_event_1_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_reps[b_idx] = torch.cat([
                    batch_local_event_2_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_1_mask_reps[b_idx] = torch.cat([
                    batch_local_event_1_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_mask_reps[b_idx] = torch.cat([
                    batch_local_event_2_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_1_dists[b_idx] = torch.cat([
                    batch_event_1_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_2_dists[b_idx] = torch.cat([
                    batch_event_2_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_1_subtypes[b_idx] += [0] * pad_length
                batch_local_event_2_subtypes[b_idx] += [0] * pad_length
                batch_coref_labels[b_idx] += [0] * pad_length
                batch_event_mask[b_idx] += [0] * pad_length
        else:
            for local_event_reps, local_event_mask_reps, event_dists in zip(
                batch_local_event_reps, batch_local_event_mask_reps, batch_event_dists
            ):
                event_num = local_event_mask_reps.size()[0]
                event_1_idx, event_2_idx = [], []
                for i in range(event_num - 1):
                    for j in range(i + 1, event_num):
                        event_1_idx.append(i)
                        event_2_idx.append(j)
                max_len = max(max_len, len(event_1_idx))
                batch_local_event_1_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_reps.append(
                    torch.index_select(local_event_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_local_event_1_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_local_event_2_mask_reps.append(
                    torch.index_select(local_event_mask_reps, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_1_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_1_idx).to(self.use_device))
                )
                batch_event_2_dists.append(
                    torch.index_select(event_dists, 0, torch.tensor(event_2_idx).to(self.use_device))
                )
                batch_event_mask.append([1] * len(event_1_idx))
            # padding
            for b_idx in range(len(batch_event_mask)):
                pad_length = max_len - len(batch_event_mask[b_idx]) if max_len > 0 else 1
                batch_local_event_1_reps[b_idx] = torch.cat([
                    batch_local_event_1_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_reps[b_idx] = torch.cat([
                    batch_local_event_2_reps[b_idx], 
                    torch.zeros((pad_length, self.encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_1_mask_reps[b_idx] = torch.cat([
                    batch_local_event_1_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_local_event_2_mask_reps[b_idx] = torch.cat([
                    batch_local_event_2_mask_reps[b_idx], 
                    torch.zeros((pad_length, self.mention_encoder_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_1_dists[b_idx] = torch.cat([
                    batch_event_1_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_2_dists[b_idx] = torch.cat([
                    batch_event_2_dists[b_idx], 
                    torch.zeros((pad_length, self.dist_dim)).to(self.use_device)
                ], dim=0).unsqueeze(0)
                batch_event_mask[b_idx] += [0] * pad_length
        batch_mask = torch.tensor(batch_event_mask).to(self.use_device)
        batch_local_event_1_reps = torch.cat(batch_local_event_1_reps, dim=0)
        batch_local_event_2_reps = torch.cat(batch_local_event_2_reps, dim=0)
        # predict event subtype
        batch_local_event_1_mask_reps = torch.cat(batch_local_event_1_mask_reps, dim=0)
        batch_local_event_2_mask_reps = torch.cat(batch_local_event_2_mask_reps, dim=0)
        event_1_subtypes_logits = self.subtype_classifier(batch_local_event_1_mask_reps)
        event_2_subtypes_logits = self.subtype_classifier(batch_local_event_2_mask_reps)
        # generate event topics
        batch_event_1_dists = torch.cat(batch_event_1_dists, dim=0)
        batch_event_2_dists = torch.cat(batch_event_2_dists, dim=0)
        loss_topic, batch_e1_topics, batch_e2_topics = self.topic_model(batch_event_1_dists, batch_event_2_dists, batch_mask)
        # matching & predict coref
        batch_event_1_reps = torch.cat([batch_local_event_1_reps, batch_local_event_1_mask_reps, batch_e1_topics], dim=-1)
        batch_event_2_reps = torch.cat([batch_local_event_2_reps, batch_local_event_2_mask_reps, batch_e2_topics], dim=-1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        # calculate loss 
        loss, batch_labels = None, None
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
            batch_labels = torch.tensor(batch_coref_labels).to(self.use_device)
            active_labels = batch_labels.view(-1)[active_loss]
            active_e_1_sutype_logits = event_1_subtypes_logits.view(-1, self.num_subtypes)[active_loss]
            active_e_2_sutype_logits = event_2_subtypes_logits.view(-1, self.num_subtypes)[active_loss]
            active_subtype_logits = torch.cat([active_e_1_sutype_logits, active_e_2_sutype_logits], dim=0)
            batch_event_1_subtypes = torch.tensor(batch_local_event_1_subtypes).to(self.use_device)
            batch_event_2_subtypes = torch.tensor(batch_local_event_2_subtypes).to(self.use_device)
            active_e_1_subtypes = batch_event_1_subtypes.view(-1)[active_loss]
            active_e_2_subtypes = batch_event_2_subtypes.view(-1)[active_loss]
            active_subtype_labels = torch.cat([active_e_1_subtypes, active_e_2_subtypes], dim=0)
            
            loss_subtype = loss_fct(active_subtype_logits, active_subtype_labels)
            loss_coref = loss_fct(active_logits, active_labels)
            loss = torch.log(1 + loss_coref) + torch.log(1 + loss_subtype) + torch.log(1 + loss_topic)
        return loss, logits, batch_mask, batch_labels
