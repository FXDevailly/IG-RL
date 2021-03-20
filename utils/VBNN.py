
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers, bias, activation = F.elu, dropout = 0.5, bn = False):
        super(NN, self).__init__()
        
        self.bn = bn
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size 
        self.n_hidden_layers = n_hidden_layers
        self.bias = bias
        self.activation = activation
        self.dropout = dropout
        self.layers = nn.ModuleList()
        if self.bn :
            self.bns = nn.ModuleList()
        
        
        
        # A SINGLE MAPPING IN->OUT if n_hidden_layers = 0 (linear regression)
        if self.n_hidden_layers == 0:
            self.linear = nn.Linear(self.input_size, self.output_size, bias = self.bias)
            
        
        else:
            self.linear = nn.Linear(self.input_size, self.hidden_size, bias = self.bias)
            for _ in range(self.n_hidden_layers):
                self.layers.append(nn.Linear(self.hidden_size, self.hidden_size, bias = self.bias))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(self.hidden_size))
            self.linear_final = nn.Linear(self.hidden_size, self.output_size, bias = self.bias)

        if self.bn:
            self.bn1 = nn.BatchNorm1d(hidden_size)
        
        
        
    def forward(self, X):
        
        y = self.activation(self.linear(X))
        if self.dropout is not None:
            y = F.dropout(y, p = self.dropout)
        if self.bn:
            y = self.bn1(y)
        if self.n_hidden_layers > 0:
            for idx in range(self.n_hidden_layers):
                y = self.activation(self.layers[idx](y))
                if self.bn:
                    y = self.bns[idx](y)
                if self.dropout is not None:
                    y = F.dropout(y, p = self.dropout)
            
            y = self.linear_final(y)
                

        return y # NOTE : TRANSFORMATION LIKE SIGMOID, SOFTMAX, etc... have to be applied on the result outside of the function
        
        
        
        
        
