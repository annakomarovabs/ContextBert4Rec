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
from catalyst.contrib.datasets import MovieLens
from catalyst.utils import get_device, set_global_seed
from torch.nn.utils.rnn import pad_sequence 

set_global_seed(100)
device = get_device()

import torch
import random

from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import *



class RecSysRunner(dl.Runner):
    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss_ae", "loss_kld", "loss"]
        }

    def handle_batch(self, batch):

        if 'targets' in batch:
            x_true = batch["targets"]
            
        loss = self.model.calculate_loss(batch)

        if 'targets' in batch:
            scores = self.model.full_sort_predict(batch)
            
            self.batch.update({'targets': batch['targets'], 'logits':scores, 'inputs':scores})
        else:
            self.batch.update({"inputs": torch.zeros((30,30)),
                           "targets": torch.zeros((30,30)),
                           'logits': torch.zeros((30,30))})

        self.batch_metrics.update({"loss": loss})
        
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)
        
    def predict_batch(self, batch):
        scores = self.model.full_sort_predict(batch)
        return scores


