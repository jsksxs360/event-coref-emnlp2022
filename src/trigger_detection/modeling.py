from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LongformerPreTrainedModel, LongformerModel
from ..tools import LabelSmoothingCrossEntropy, FocalLoss, CRF
from ..tools import FullyConnectedLayer

class LongformerSoftmaxForTD(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.use_ffnn_layer = args.use_ffnn_layer
        if self.use_ffnn_layer:
            self.ffnn_size = args.ffnn_size if args.ffnn_size != -1 else config.hidden_size
            self.mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, config.hidden_dropout_prob)
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.ffnn_size if args.use_ffnn_layer else config.hidden_size, self.num_labels)
        self.loss_type = args.softmax_loss
        self.post_init()
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        if self.use_ffnn_layer:
            sequence_output = self.mlp(sequence_output)
        else:
            sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

class LongformerCrfForTD(LongformerPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.use_ffnn_layer = args.use_ffnn_layer
        if self.use_ffnn_layer:
            self.ffnn_size = args.ffnn_size if args.ffnn_size != -1 else config.hidden_size
            self.mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, config.hidden_dropout_prob)
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.ffnn_size if args.use_ffnn_layer else config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.post_init()
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        if self.use_ffnn_layer:
            sequence_output = self.mlp(sequence_output)
        else:
            sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions=logits, tags=labels, mask=attention_mask)
        return loss, logits
