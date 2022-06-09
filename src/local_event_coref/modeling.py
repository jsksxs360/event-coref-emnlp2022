import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers import RobertaPreTrainedModel, RobertaModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from torch.nn import CrossEntropyLoss
from ..tools import LabelSmoothingCrossEntropy, FocalLoss
from ..tools import SimpleTopicModel, SimpleTopicModelwithBN, SimpleTopicVMFModel

TOPIC_MODEL = {
    'stm': SimpleTopicModel, 
    'stm_bn': SimpleTopicModelwithBN, 
    'vmf': SimpleTopicVMFModel
}
COSINE_SPACE_DIM = 64
COSINE_SLICES = 128
COSINE_FACTOR = 4

class BertForPairwiseEC(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
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
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
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
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_seq_reps
    
    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits

class RobertaForPairwiseEC(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
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
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
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
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_seq_reps
    
    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return loss, logits

class BertForPairwiseECWithMask(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.num_subtypes = args.num_subtypes
        self.hidden_size = config.hidden_size
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.subtype_classifier = nn.Linear(self.hidden_size, self.num_subtypes)
        self.matching_style = args.matching_style
        if 'cosine' not in self.matching_style:
            if self.matching_style == 'base':
                multiples = 4
            elif self.matching_style == 'multi':
                multiples = 6
            self.coref_classifier = nn.Linear(multiples * self.hidden_size, self.num_labels)
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size * 2, self.cosine_space_dim)
            if self.matching_style == 'cosine':
                self.coref_classifier = nn.Linear(4 * self.hidden_size + self.cosine_slices, self.num_labels)
            elif self.matching_style == 'multi_cosine':
                self.coref_classifier = nn.Linear(6 * self.hidden_size + self.cosine_slices, self.num_labels)
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
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
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_seq_reps

    def forward(self, batch_inputs, batch_inputs_with_mask, batch_e1_idx, batch_e2_idx, labels=None, subtypes=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs_with_mask = self.bert(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask[0]
        sequence_output_with_mask = self.dropout(sequence_output_with_mask)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_event_mask_1_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e1_idx)
        batch_event_mask_2_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e2_idx)
        batch_event_mask_reps = torch.cat([batch_event_mask_1_reps, batch_event_mask_2_reps], dim=1)
        subtypes_logits = self.subtype_classifier(batch_event_mask_reps)
        batch_event_mask_1_reps = batch_event_mask_1_reps.squeeze(dim=1)
        batch_event_mask_2_reps = batch_event_mask_2_reps.squeeze(dim=1)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_event_mask_1_reps], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_event_mask_2_reps], dim=-1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            assert subtypes is not None
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            loss_coref = loss_fct(logits, labels)
            loss_subtype = loss_fct(subtypes_logits.view(-1, self.num_subtypes), subtypes.view(-1))
            loss = torch.log(1 + loss_coref) + torch.log(1 + loss_subtype)
        return loss, logits

class RobertaForPairwiseECWithMask(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.num_subtypes = args.num_subtypes
        self.hidden_size = config.hidden_size
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=self.hidden_size)
        self.subtype_classifier = nn.Linear(self.hidden_size, args.num_subtypes)
        self.matching_style = args.matching_style
        if 'cosine' not in self.matching_style:
            if self.matching_style == 'base':
                multiples = 4
            elif self.matching_style == 'multi':
                multiples = 6
            self.coref_classifier = nn.Linear(multiples * self.hidden_size, self.num_labels)
        else:
            self.cosine_space_dim, self.cosine_slices, self.tensor_factor = COSINE_SPACE_DIM, COSINE_SLICES, COSINE_FACTOR
            self.cosine_mat_p = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_slices), requires_grad=True))
            self.cosine_mat_q = nn.Parameter(torch.rand((self.tensor_factor, self.cosine_space_dim), requires_grad=True))
            self.cosine_ffnn = nn.Linear(self.hidden_size * 2, self.cosine_space_dim)
            if self.matching_style == 'cosine':
                self.coref_classifier = nn.Linear(4 * self.hidden_size + self.cosine_slices, self.num_labels)
            elif self.matching_style == 'multi_cosine':
                self.coref_classifier = nn.Linear(6 * self.hidden_size + self.cosine_slices, self.num_labels)
        self.post_init()

    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
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
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_seq_reps
    
    def forward(self, batch_inputs, batch_inputs_with_mask, batch_e1_idx, batch_e2_idx, labels=None, subtypes=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs_with_mask = self.roberta(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask[0]
        sequence_output_with_mask = self.dropout(sequence_output_with_mask)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_event_mask_1_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e1_idx)
        batch_event_mask_2_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e2_idx)
        batch_event_mask_reps = torch.cat([batch_event_mask_1_reps, batch_event_mask_2_reps], dim=1)
        subtypes_logits = self.subtype_classifier(batch_event_mask_reps)
        batch_event_mask_1_reps = batch_event_mask_1_reps.squeeze(dim=1)
        batch_event_mask_2_reps = batch_event_mask_2_reps.squeeze(dim=1)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_event_mask_1_reps], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_event_mask_2_reps], dim=-1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            assert subtypes is not None
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            loss_coref = loss_fct(logits, labels)
            loss_subtype = loss_fct(subtypes_logits.view(-1, self.num_subtypes), subtypes.view(-1))
            loss = torch.log(1 + loss_coref) + torch.log(1 + loss_subtype)
        return loss, logits

class BertForPairwiseECwithTopic(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size + args.topic_dim
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.topic_model = TOPIC_MODEL[args.topic_model](args=args)
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
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
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
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_seq_reps
    
    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, batch_e1_dists, batch_e2_dists, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        # generate event topics
        batch_event_1_dists = batch_e1_dists.unsqueeze(dim=1)
        batch_event_2_dists = batch_e2_dists.unsqueeze(dim=1)
        loss_topic, batch_e1_topics, batch_e2_topics = self.topic_model(batch_event_1_dists, batch_event_2_dists)
        batch_e1_topics = batch_e1_topics.squeeze(dim=1)
        batch_e2_topics = batch_e2_topics.squeeze(dim=1)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_e1_topics], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_e2_topics], dim=-1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            loss_coref = loss_fct(logits, labels)
            loss = torch.log(1 + loss_coref) + torch.log(1 + loss_topic)
        return loss, logits

class RobertaForPairwiseECwithTopic(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.hidden_size = config.hidden_size + args.topic_dim
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.topic_model = TOPIC_MODEL[args.topic_model](args=args)
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
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
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
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_seq_reps
    
    def forward(self, batch_inputs, batch_e1_idx, batch_e2_idx, batch_e1_dists, batch_e2_dists, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        # generate event topics
        batch_event_1_dists = batch_e1_dists.unsqueeze(dim=1)
        batch_event_2_dists = batch_e2_dists.unsqueeze(dim=1)
        loss_topic, batch_e1_topics, batch_e2_topics = self.topic_model(batch_event_1_dists, batch_event_2_dists)
        batch_e1_topics = batch_e1_topics.squeeze(dim=1)
        batch_e2_topics = batch_e2_topics.squeeze(dim=1)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_e1_topics], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_e2_topics], dim=-1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            loss_coref = loss_fct(logits, labels)
            loss = torch.log(1 + loss_coref) + torch.log(1 + loss_topic)
        return loss, logits

class BertForPairwiseECwithMaskTopic(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.num_subtypes = args.num_subtypes
        self.hidden_size = 2 * config.hidden_size + args.topic_dim
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.subtype_classifier = nn.Linear(config.hidden_size, self.num_subtypes)
        self.topic_model = TOPIC_MODEL[args.topic_model](args=args)
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
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
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
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_seq_reps
    
    def forward(self, batch_inputs, batch_inputs_with_mask, batch_e1_idx, batch_e2_idx, batch_e1_dists, batch_e2_dists, labels=None, subtypes=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs_with_mask = self.bert(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask[0]
        sequence_output_with_mask = self.dropout(sequence_output_with_mask)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_event_mask_1_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e1_idx)
        batch_event_mask_2_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e2_idx)
        batch_event_mask_reps = torch.cat([batch_event_mask_1_reps, batch_event_mask_2_reps], dim=1)
        subtypes_logits = self.subtype_classifier(batch_event_mask_reps)
        batch_event_mask_1_reps = batch_event_mask_1_reps.squeeze(dim=1)
        batch_event_mask_2_reps = batch_event_mask_2_reps.squeeze(dim=1)
        # generate event topics
        batch_event_1_dists = batch_e1_dists.unsqueeze(dim=1)
        batch_event_2_dists = batch_e2_dists.unsqueeze(dim=1)
        loss_topic, batch_e1_topics, batch_e2_topics = self.topic_model(batch_event_1_dists, batch_event_2_dists)
        batch_e1_topics = batch_e1_topics.squeeze(dim=1)
        batch_e2_topics = batch_e2_topics.squeeze(dim=1)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_event_mask_1_reps, batch_e1_topics], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_event_mask_2_reps, batch_e2_topics], dim=-1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            loss_coref = loss_fct(logits, labels)
            loss_subtype = loss_fct(subtypes_logits.view(-1, self.num_subtypes), subtypes.view(-1))
            loss = torch.log(1 + loss_coref) + torch.log(1 + loss_subtype) + torch.log(1 + loss_topic)
        return loss, logits

class RobertaForPairwiseECwithMaskTopic(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.num_subtypes = args.num_subtypes
        self.hidden_size = 2 * config.hidden_size + args.topic_dim
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.subtype_classifier = nn.Linear(config.hidden_size, self.num_subtypes)
        self.topic_model = TOPIC_MODEL[args.topic_model](args=args)
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
        self.post_init()
    
    def _multi_cosine(self, batch_event_1_reps, batch_event_2_reps):
        batch_event_1_reps = self.cosine_ffnn(batch_event_1_reps)
        batch_event_1_reps = batch_event_1_reps.unsqueeze(dim=1)
        batch_event_1_reps = self.cosine_mat_q * batch_event_1_reps
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        batch_event_1_reps = torch.matmul(batch_event_1_reps, self.cosine_mat_p)
        batch_event_1_reps = batch_event_1_reps.permute((0, 2, 1))
        # vector normalization
        norms_1 = (batch_event_1_reps ** 2).sum(axis=-1, keepdims=True) ** 0.5
        batch_event_1_reps = batch_event_1_reps / norms_1
        
        batch_event_2_reps = self.cosine_ffnn(batch_event_2_reps)
        batch_event_2_reps = batch_event_2_reps.unsqueeze(dim=1)
        batch_event_2_reps = self.cosine_mat_q * batch_event_2_reps
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
        batch_event_2_reps = torch.matmul(batch_event_2_reps, self.cosine_mat_p)
        batch_event_2_reps = batch_event_2_reps.permute((0, 2, 1))
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
        elif self.matching_style == 'cosine':
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_multi_cosine], dim=-1)
        elif self.matching_style == 'multi_cosine':
            batch_e1_e2_multi = batch_event_1_reps * batch_event_2_reps
            batch_multi_cosine = self._multi_cosine(batch_event_1_reps, batch_event_2_reps)
            batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2_multi, batch_multi_cosine], dim=-1)
        return batch_seq_reps
    
    def forward(self, batch_inputs, batch_inputs_with_mask, batch_e1_idx, batch_e2_idx, batch_e1_dists, batch_e2_dists, labels=None, subtypes=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs_with_mask = self.roberta(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask[0]
        sequence_output_with_mask = self.dropout(sequence_output_with_mask)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_event_mask_1_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e1_idx)
        batch_event_mask_2_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e2_idx)
        batch_event_mask_reps = torch.cat([batch_event_mask_1_reps, batch_event_mask_2_reps], dim=1)
        subtypes_logits = self.subtype_classifier(batch_event_mask_reps)
        batch_event_mask_1_reps = batch_event_mask_1_reps.squeeze(dim=1)
        batch_event_mask_2_reps = batch_event_mask_2_reps.squeeze(dim=1)
        # generate event topics
        batch_event_1_dists = batch_e1_dists.unsqueeze(dim=1)
        batch_event_2_dists = batch_e2_dists.unsqueeze(dim=1)
        loss_topic, batch_e1_topics, batch_e2_topics = self.topic_model(batch_event_1_dists, batch_event_2_dists)
        batch_e1_topics = batch_e1_topics.squeeze(dim=1)
        batch_e2_topics = batch_e2_topics.squeeze(dim=1)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_event_mask_1_reps, batch_e1_topics], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_event_mask_2_reps, batch_e2_topics], dim=-1)
        batch_seq_reps = self._matching_func(batch_event_1_reps, batch_event_2_reps)
        logits = self.coref_classifier(batch_seq_reps)
        loss = None
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            loss_coref = loss_fct(logits, labels)
            loss_subtype = loss_fct(subtypes_logits.view(-1, self.num_subtypes), subtypes.view(-1))
            loss = torch.log(1 + loss_coref) + torch.log(1 + loss_subtype) + torch.log(1 + loss_topic)
        return loss, logits
