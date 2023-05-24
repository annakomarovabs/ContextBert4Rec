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

from BERT4Rec import BERT4Rec


class ContextBERT4Rec(BERT4Rec):

    def __init__(self, n_items, hidden_size, mask_ratio):
        super(BERT4Rec, self).__init__()
        
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

        # load dataset info
        self.n_items = n_items
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.weekday_embedding = nn.Embedding(7, self.hidden_size)
        self.hours_embedding = nn.Embedding(24, self.hidden_size)
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



    def reconstruct_test_data(self,
                              item_seq,
                              item_seq_len,
                              dow,
                              hours,
                              dow_valid,
                              hours_valid,
                              particular_day=-1,
                              ):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  
        dow = torch.cat((dow, padding.unsqueeze(-1)), dim=-1)
        hours = torch.cat((hours, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
            if particular_day == -1:
                dow[batch_id][last_position] = dow_valid[batch_id]
                hours[batch_id][last_position] = hours_valid[batch_id]
            else:
                dow[batch_id][last_position] = particular_day
                hours[batch_id][last_position] = particular_day
        return item_seq, dow, hours

    def forward(self, item_seq, dow, hours, return_explanations=False):
        
        
        dow_embeddings = self.weekday_embedding(dow.long())
        hours_embeddings = self.hours_embedding(hours.long())
        
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding + dow_embeddings + hours_embeddings
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


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ].long()
        masked_item_seq, pos_items, neg_items, masked_index = self.reconstruct_train_data(item_seq)

        seq_output = self.forward(masked_item_seq, dow=interaction['dow'], hours=interaction['hours'])
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


    def full_sort_predict(self, 
                          interaction,
                          return_explanations=False,
                          particular_day=-1):
        
        item_seq = interaction[self.ITEM_SEQ].long()
        item_seq_len = interaction[self.ITEM_SEQ_LEN].long()
        item_seq, dow, hours = self.reconstruct_test_data(item_seq,
                                              item_seq_len,
                                              dow=interaction['dow'],
                                              hours=interaction['hours'],
                                              dow_valid=interaction['dow_valid'].long(),
                                              hours_valid=interaction['hours_valid'].long(),
                                              particular_day=particular_day)
        
        
        if return_explanations:
            seq_output, expl = self.forward(item_seq,
                                            dow=dow,
                                            hours=hours,
                                            return_explanations=return_explanations)
        else:
            seq_output = self.forward(item_seq,
                                      dow=dow,
                                      hours=hours,
                                      return_explanations=return_explanations)
            
        
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

    
    
