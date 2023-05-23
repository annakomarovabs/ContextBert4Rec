import pandas as pd
import numpy as np
from typing import List

import catalyst 
import recbole

from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from catalyst import dl, metrics
from catalyst.utils import get_device, set_global_seed
from torch.nn.utils.rnn import pad_sequence 

set_global_seed(100)
device = get_device()

import torch
import random

from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import *

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.
    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, return_explanations=False):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        if return_explanations:
            return hidden_states, attention_probs
        else:
            return hidden_states

        
        

class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask,return_explanations=False):
        
        if return_explanations:
            attention_output, expl = self.multi_head_attention(hidden_states, attention_mask,
                                                         return_explanations=return_explanations)
            
        else:
            attention_output = self.multi_head_attention(hidden_states, attention_mask,
                                                         return_explanations=return_explanations)
        feedforward_output = self.feed_forward(attention_output)
        
        if return_explanations:
            return feedforward_output, expl
        else:
            return feedforward_output
    
    
    
class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.
    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True,
                return_explanations=False):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for idx, layer_module in enumerate(self.layer):
            
            if return_explanations:
                hidden_states, expl = layer_module(hidden_states, attention_mask, 
                                         return_explanations=return_explanations)
            else:            
                hidden_states = layer_module(hidden_states, attention_mask, 
                                             return_explanations=return_explanations)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        if return_explanations:
            return all_encoder_layers, expl
        else:
            return all_encoder_layers

    

    
class BERT4Rec(torch.nn.Module):

    def __init__(self, n_items, hidden_size, mask_ratio):
        super(BERT4Rec, self).__init__()

        # load parameters info
        self.n_layers = 2
        self.n_heads = 2
        self.hidden_size = hidden_size  
        self.inner_size = 128 
        self.hidden_dropout_prob = 0.2
        self.attn_dropout_prob = 0.2
        self.hidden_act = 'sigmoid'
        self.layer_norm_eps = 1e-5
        self.ITEM_SEQ = 'seq_i'
        self.ITEM_SEQ_LEN = 'seq_len'
        self.max_seq_length = 200
        

        self.mask_ratio = mask_ratio

        self.loss_type =  'CE'
        self.initializer_range = 1e-2

        self.n_items = n_items
        self.mask_token = self.n_items 
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0) 
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size) 
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        try:
            assert self.loss_type in ['BPR', 'CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)
        
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) 
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def _neg_sample(self, item_set):
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:] 
        return sequence

    def reconstruct_train_data(self, item_seq):
        """
        Mask item sequence for training.
        """
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()

        masked_item_sequence = []
        pos_items = []
        neg_items = []
        masked_index = []
        for instance in sequence_instances:
            masked_sequence = instance.copy()
            pos_item = []
            neg_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                if item == 0:
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    neg_item.append(self._neg_sample(instance))
                    masked_sequence[index_id] = self.mask_token
                    index_ids.append(index_id)

            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            neg_items.append(self._padding_sequence(neg_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))

        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence, pos_items, neg_items, masked_index

    def reconstruct_test_data(self, item_seq, item_seq_len):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        return item_seq

    def forward(self, item_seq, return_explanations=False):
        
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        if return_explanations:
            trm_output, explanations = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True,
                                         return_explanations=return_explanations)
        else:
            trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True,
                                         return_explanations=return_explanations)
            
        output = trm_output[-1]
        
        if return_explanations:
            return output, explanations
        else:
            return output

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.
        Examples:
            sequence: [1 2 3 4 5]
            masked_sequence: [1 mask 3 mask 5]
            masked_index: [1, 3]
            max_length: 5
            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].long()
        masked_item_seq, pos_items, neg_items, masked_index = self.reconstruct_train_data(item_seq)

        seq_output = self.forward(masked_item_seq)
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1)) 
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1) 
        seq_output = torch.bmm(pred_index_map, seq_output)  

        if self.loss_type == 'BPR':
            pos_items_emb = self.item_embedding(pos_items) 
            neg_items_emb = self.item_embedding(neg_items) 
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1) 
            targets = (masked_index > 0).float()
            loss = - torch.sum(torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets) \
                   / torch.sum(targets)
            return loss

        elif self.loss_type == 'CE':
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            test_item_emb = self.item_embedding.weight[:self.n_items] 
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  
            targets = (masked_index > 0).float().view(-1)  

            loss = torch.sum(loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) \
                   / torch.sum(targets)
            return loss
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")


    def full_sort_predict(self, interaction, return_explanations=False):
        item_seq = interaction[self.ITEM_SEQ].long()
        item_seq_len = interaction[self.ITEM_SEQ_LEN].long()
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        
        if return_explanations:
            seq_output, expl = self.forward(item_seq, return_explanations=return_explanations)
        else:
            seq_output = self.forward(item_seq, return_explanations=return_explanations)
            
        
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1) 
        test_items_emb = self.item_embedding.weight[:self.n_items] 
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  
                
        idxs = item_seq.nonzero()
        item_seq[item_seq==self.n_items] = 0
        scores[idxs[:,0], item_seq[idxs[:,0],idxs[:,1]].long()] = -1000

        if return_explanations:
            return scores, expl
        else:
            return scores

    
    
