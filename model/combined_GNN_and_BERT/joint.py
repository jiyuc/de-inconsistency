from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F


class JointModelForSequenceClassification(BertPreTrainedModel):
    """
    Joint Modelling of GO-PMIDgene semantic relationship using standard
    BERT word vsm and pre-trained GO BG-KG.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(16, 16, bias=True)
        self.classifier = nn.Linear(config.hidden_size + 16, config.num_labels)  # |u,v, u*v, u-v|
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            node_embedding=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        """
        # normalise
        # pooled_output = F.normalize(pooled_output, p=2.0, dim=1, eps=1e-12, out=None)
        go_hier_encodings = self.linear(go_hier_encodings)  # expand 16dim GO to 768 dim
        """
        # flat concatenation as joint embedding |u||v|
        node_embedding = torch.randn(node_embedding.shape)
        node_embedding = self.linear(node_embedding)
        concatenated_output = torch.cat([pooled_output, node_embedding], dim=1)
        """
        # element_wise dot product |u*v|
        e_dot = pooled_output * go_hier_encodings

        # absolute element_wise difference |u-v|
        d_dot = pooled_output - go_hier_encodings

        concatenated_output = torch.cat([concatenated_output, e_dot, d_dot], dim=1)

        # forward into a linear layer
        #forwarded_output = self.linear(concatenated_output)
        """

        logits = self.classifier(concatenated_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
