# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 16:19:39 2019

@author: chenkkkk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import CrossNet, DNN


class DCN(BaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, embedding_size=8,cross_num=2,
                 dnn_hidden_units=(128, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation=F.relu, dnn_use_bn=False, task='binary', device='cpu'):

        super(DCN, self).__init__(linear_feature_columns, dnn_feature_columns, embedding_size=embedding_size,
                                     dnn_hidden_units=dnn_hidden_units,
                                     l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, init_std=init_std,
                                     seed=seed,
                                     dnn_dropout=dnn_dropout, dnn_activation=dnn_activation,
                                     task=task, device=device)
        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = cross_num
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, embedding_size, ), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, init_std=init_std)
        self.dnn_linear = nn.Linear(self.compute_input_dim(dnn_feature_columns, embedding_size, )+dnn_hidden_units[-1], 1, bias=False)
        self.crossnet = CrossNet(input_feature_num = self.compute_input_dim(dnn_feature_columns, embedding_size, ),layer_num=cross_num, seed=1024)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        
        dnn_input = combined_dnn_input(sparse_embedding_list,dense_value_list)
        
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:  # Deep & Cross
            deep_out = self.dnn(dnn_input)
            cross_out = self.crossnet(dnn_input)
            stack_out = torch.cat((cross_out, deep_out),dim=-1)
            final_logit = self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            final_logit = self.dnn_linear(deep_out)
        elif self.cross_num > 0:  # Only Cross
            cross_out = self.crossnet(dnn_input)
            final_logit = self.dnn_linear(cross_out)
        else:  # Error
            raise NotImplementedError
        
        

        y_pred = self.out(final_logit)
        return y_pred