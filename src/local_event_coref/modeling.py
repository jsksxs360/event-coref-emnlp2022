import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import LongformerPreTrainedModel, LongformerModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from torch.nn import CrossEntropyLoss
from ..tools import LabelSmoothingCrossEntropy, FocalLoss

class BertForPairwiseEC(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.coref_classifier = nn.Linear(3 * config.hidden_size, args.num_labels)
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.post_init()
    
    def forward(self, input_ids, attention_mask, token_type_ids, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
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
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.coref_classifier = nn.Linear(3 * config.hidden_size, args.num_labels)
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.post_init()
    
    def forward(self, input_ids, attention_mask, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
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

class LongformerForPairwiseEC(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.coref_classifier = nn.Linear(3 * config.hidden_size, args.num_labels)
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.post_init()
    
    def forward(self, input_ids, attention_mask, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
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
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.coref_classifier = nn.Linear(6 * config.hidden_size, args.num_labels)
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.post_init()
    
    def forward(self, batch_inputs, batch_inputs_with_mask, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.bert(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs_with_mask = self.bert(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask[0]
        sequence_output_with_mask = self.dropout(sequence_output_with_mask)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_event_mask_1_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e1_idx).squeeze(dim=1)
        batch_event_mask_2_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e2_idx).squeeze(dim=1)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_event_mask_1_reps], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_event_mask_2_reps], dim=-1)
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
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

class RobertaForPairwiseECWithMask(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.coref_classifier = nn.Linear(6 * config.hidden_size, args.num_labels)
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.post_init()
    
    def forward(self, batch_inputs, batch_inputs_with_mask, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.roberta(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs_with_mask = self.roberta(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask[0]
        sequence_output_with_mask = self.dropout(sequence_output_with_mask)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_event_mask_1_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e1_idx).squeeze(dim=1)
        batch_event_mask_2_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e2_idx).squeeze(dim=1)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_event_mask_1_reps], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_event_mask_2_reps], dim=-1)
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
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

class LongformerForPairwiseECWithMask(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.coref_classifier = nn.Linear(6 * config.hidden_size, args.num_labels)
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.post_init()
    
    def forward(self, batch_inputs, batch_inputs_with_mask, batch_e1_idx, batch_e2_idx, labels=None):
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs_with_mask = self.longformer(**batch_inputs_with_mask)
        sequence_output_with_mask = outputs_with_mask[0]
        sequence_output_with_mask = self.dropout(sequence_output_with_mask)
        # extract events
        batch_event_1_reps = self.span_extractor(sequence_output, batch_e1_idx).squeeze(dim=1)
        batch_event_2_reps = self.span_extractor(sequence_output, batch_e2_idx).squeeze(dim=1)
        batch_event_mask_1_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e1_idx).squeeze(dim=1)
        batch_event_mask_2_reps = self.mask_span_extractor(sequence_output_with_mask, batch_e2_idx).squeeze(dim=1)
        batch_event_1_reps = torch.cat([batch_event_1_reps, batch_event_mask_1_reps], dim=-1)
        batch_event_2_reps = torch.cat([batch_event_2_reps, batch_event_mask_2_reps], dim=-1)
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
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

class BertForPairwiseECWithMaskAndSubtype(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_subtypes = args.num_subtypes
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.subtype_classifier = nn.Linear(config.hidden_size, args.num_subtypes)
        self.coref_classifier = nn.Linear(6 * config.hidden_size, args.num_labels)
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.post_init()
    
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
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
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
            loss = 0.5 * loss_coref + 0.5 * loss_subtype
        return loss, logits

class RobertaForPairwiseECWithMaskAndSubtype(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_subtypes = args.num_subtypes
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.subtype_classifier = nn.Linear(config.hidden_size, args.num_subtypes)
        self.coref_classifier = nn.Linear(6 * config.hidden_size, args.num_labels)
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.post_init()
    
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
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
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
            loss = 0.5 * loss_coref + 0.5 * loss_subtype
        return loss, logits

class LongformerForPairwiseECWithMaskAndSubtype(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_subtypes = args.num_subtypes
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.mask_span_extractor = SelfAttentiveSpanExtractor(input_dim=config.hidden_size)
        self.subtype_classifier = nn.Linear(config.hidden_size, args.num_subtypes)
        self.coref_classifier = nn.Linear(6 * config.hidden_size, args.num_labels)
        self.loss_type = args.softmax_loss
        self.use_device = args.device
        self.post_init()
    
    def forward(self, batch_inputs, batch_inputs_with_mask, batch_e1_idx, batch_e2_idx, labels=None, subtypes=None):
        outputs = self.longformer(**batch_inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs_with_mask = self.longformer(**batch_inputs_with_mask)
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
        batch_e1_e2 = batch_event_1_reps * batch_event_2_reps
        batch_seq_reps = torch.cat([batch_event_1_reps, batch_event_2_reps, batch_e1_e2], dim=-1)
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
            loss = 0.5 * loss_coref + 0.5 * loss_subtype
        return loss, logits
