import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
import collections
import logging
import datetime
import numpy as np
import time
import traci
import pickle
import networkx as nx
import dgl
from torch.utils.data import DataLoader
import sys
import torch
import copy
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix
import pylab as pl
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from statistics import mean 

def comput(request_ends, comput_model_queue, baseline_reward_queue = None, greedy_reward_queue = None,  tested_learner_ends = None, tested = None, reward_queues = None):
    device = "cuda:0"
    env_params = comput_model_queue.get() #
    if env_params.additional_params['mode'] == 'train' or not env_params.additional_params['sequential_computation']:
        model = comput_model_queue.get() 
        model.to(device) 
        if env_params.additional_params['Policy_Type'] == "Q_Learning":
            model.eval()
    else:
        models = collections.OrderedDict()
        for idx, request_end in request_ends.items():
            model = request_end.recv()
            if idx == 0:
                env_params = env_params
            if type(model) != str :
                model.to(device)
                if env_params.additional_params['Policy_Type'] == "Q_Learning":
                    model.eval()
                models[idx] = model

    counter = 0 

    
    
    if env_params.additional_params['Policy_Type'] == "Q_Learning":
        with torch.no_grad():    
            while True:
 
                if env_params.additional_params['mode'] == "train":
                    # UPDATE MODEL
                    if counter % env_params.additional_params['update_comput_model_frequency'] == 0 :
                        if not comput_model_queue.empty():
                            try:
                                model = comput_model_queue.get_nowait() # UPDATE MODEL EVERY TIME 
                                model.eval()
                                model.to(device)
                                #print("UPDATE MODEL")
                            except:
                                pass



                # GET THE REQUEST
                compute_idx = []
                random_idx = []
                graphs_list = []
                lengths_list = []
                batched_state = []
                actions_sizes = []
                
                

                if env_params.additional_params['mode'] == 'train' or not env_params.additional_params['sequential_computation']:   

                    for idx, request_end in request_ends.copy().items():
                        request = request_end.recv()
                        if type(request) == str :
                            if request == 'Done':
                                del request_ends[idx]
                            else:
                                random_idx.append(idx)
                        else:
                            if env_params.additional_params['GCN']:                        
                                graphs_list.append(request[0])
                            elif not env_params.additional_params['GCN']:
                                batched_state.append(request[0])
                            lengths_list.append(request[1])
                            actions_sizes.append(request[2])
                            compute_idx.append(idx)


                    if graphs_list or batched_state: # CHECK THAT IT'S NOT EMPTY

                        if env_params.additional_params['GCN']:
                            n_graphs = len(graphs_list)
                            batched_graph = dgl.batch(graphs_list, node_attrs = ['state', 'node_type'], edge_attrs = ['rel_type', 'norm'])
                            graphs_list = []

                            s = time.time()
                            hid, Q_values = model.forward(batched_graph, device, testing = True if env_params.additional_params['mode'] == 'test' else False, actions_sizes = torch.cat(tuple(actions_sizes),dim=0))
                            if env_params.additional_params['gaussian_mixture']:
                                Q_values = sample(Q_values[0],Q_values[1],Q_values[2])
                            if env_params.additional_params['policy'] == 'binary':
                                _ , Q_values = torch.max(Q_values, dim =1)
                                Q_values = Q_values.split(lengths_list, dim = 0)
                            else:
                                if len(lengths_list) == 1:
                                    Q_values = [Q_values]
                                else:
                                    Q_values = Q_values.split(lengths_list, dim = 0)
                        elif not env_params.additional_params['GCN']:
                            n = len(batched_state)
                            _ , Q_values = model.forward(batched_state, device, testing = True if env_params.additional_params['mode'] =='test' else False, actions_sizes = torch.cat(tuple(actions_sizes),dim=0))
                            
                            
                            if env_params.additional_params['policy'] == 'binary':
                                _ , Q_values = torch.max(Q_values, dim =1)
                            Q_values = Q_values.view(n, model.n_tls)           
                    # SEND RANDOM FOR OTHERS
                    for idx in random_idx:
                        request_ends[idx].send('N/A')

                    # SEND COMPUTED RESULT
                    for result_idx, request_idx in enumerate(compute_idx):
                        request_ends[request_idx].send(Q_values[result_idx].squeeze().cpu().numpy())             



                            
                 
                # WHEN USING SEQUENTIAL COMPUTATION 
                else : 
                    for idx, request_end in request_ends.copy().items():
                        request = request_end.recv()
                        if type(request) == str :
                            if request == 'Done':
                                del request_ends[idx]
                            else:
                                request_end.send('N/A')
                        else:                                          
                            s = time.time()
                            if env_params.additional_params['GCN']:
                                hid, Q_values = models[idx].forward(request[0], device, testing = True if env_params.additional_params['mode'] == 'test' else False, actions_sizes = request[2])

                                if env_params.additional_params['policy'] == 'binary':
                                    _ , Q_values = torch.max(Q_values, dim =1)

                            elif not env_params.additional_params['GCN']:
                                _ , Q_values = models[idx].forward(request[0], device, testing = True if env_params.additional_params['mode'] =='test' else False, actions_sizes = request[2])
                                if env_params.additional_params['policy'] == 'binary':
                                    _ , Q_values = torch.max(Q_values, dim =1)

                            request_end.send(Q_values.view(1,-1).squeeze().cpu().numpy())
