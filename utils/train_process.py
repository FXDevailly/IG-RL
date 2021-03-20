"""Contains an experiment class for running simulations."""
import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
import collections
import logging
import datetime
import numpy as np
import time
import traci
from utils.model import *
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
from statistics import mean 
from torch.autograd import Variable
from utils.rl_tools import *
from utils.Memory import *


def train(memory_queues, learn_model_queue, comput_model_queue, reward_queues, baseline_reward_queue, greedy_reward_queue, tested_ends = None, tested = None, num = None):
    random.seed(a=0) 
    # INITIALIZE MODEL
    device = "cuda:1"
    model = learn_model_queue.get() # WAIT FOR MODEL
    model.to(device)
    
    
    # INITIALIZE TARGET MODEL
    if model.env_params.additional_params['target_model_update_frequency'] is not None:
        target_model = copy.deepcopy(model)
        target_model.eval()
        def update_target_graph():    
            target_model.load_state_dict(model.state_dict())
        

    # DEFINE PATH TO SAVE TRAINING RESULTS
    writer = SummaryWriter(model.env_params.additional_params["tb_foldername"])
    
    
    # MULTILINE CHARTS FOR TENSORBOARD
    if model.env_params.additional_params['mode'] == 'train':
        writer.add_custom_scalars_multilinechart(['general_train_loss', 'general_test_loss'], title = 'General Train/Test Loss')
    elif model.env_params.additional_params['mode'] == 'test':
        writer.add_custom_scalars_multilinechart([str(x) for x in tested], title = 'Rewards Comparison')        

        
    # DEFINE ALL HYPERPARAMETERS TO BE REGISTERED IN TENSORBOARD
    for k,v in model.env_params.additional_params.items():
        if type(v) == list:
            for idx, elem in enumerate(v):
                if idx == 0:
                    value = str(elem)
                else:
                    value += str(" ||  " + str(elem))

            writer.add_text(k, value, global_step=None, walltime=None)
        else:
            writer.add_text(k, str(v), global_step=None, walltime=None)

            
    
    # AGGREGATION FUNCTION USED IN DATALOADER (TO CREATE BATCHES)
    def collate(samples):
        try:
            batched_graph, imm_rewards, actions, choices, forced_mask, forced_value, actions_sizes, future_rewards, next_batched_graph, prev_imm_rewards, next_choices, next_forced_mask, next_forced_value, next_actions_sizes =  map(merge_, zip(*samples)) # global_rewards # choices
        except Exception as e: 
            print(e)
            pass
        if model.env_params.additional_params['GCN']:
            batched_graph = dgl.batch(batched_graph, node_attrs = ['state', 'node_type'], edge_attrs = ['rel_type', 'norm'])
            next_batched_graph = dgl.batch(next_batched_graph, node_attrs = ['state', 'node_type'], edge_attrs = ['rel_type', 'norm'])
        return batched_graph, torch.FloatTensor(imm_rewards, ).view(-1).type(torch.float), torch.LongTensor(actions).view(-1), torch.ByteTensor(choices).view(-1), torch.LongTensor(forced_mask).view(-1), torch.LongTensor(forced_value).view(-1), torch.LongTensor(actions_sizes).view(-1), torch.FloatTensor(future_rewards).view(-1), next_batched_graph, torch.FloatTensor(prev_imm_rewards).view(-1), torch.LongTensor(next_choices).view(-1), torch.LongTensor(next_forced_mask).view(-1), torch.LongTensor(next_forced_value).view(-1), torch.LongTensor(next_actions_sizes).view(-1) #, # choices torch.FloatTensor(global_rewards).view(-1).type(torch.float)
    
    
    # FUNCTION USED IN COLLATE 
    def merge_(*items):
        a = []
        if model.env_params.additional_params['GCN']:
            try:
                for item in items[0]:
                    a.extend(item)
            except:
                for item in items:
                    a.extend(item)
        elif not model.env_params.additional_params['GCN']:
            try: 
                items[0][0][0][0]
                try:
                    for item in items[0]:
                        a.append(item)
                except:
                    for item in items:
                        a.append(item)
            except:
                try:
                    for item in items[0]:
                        a.extend(item)
                except:
                    for item in items:
                        a.extend(item)
        return a
    
    def split_list(a_list):
        half = len(a_list)//2
        return a_list[:half], a_list[half:]    


    # INITIALIZE MEMORY BUFFERS
    if model.env_params.additional_params['separate_memory_buffers']:
        train_memory_buffers = collections.OrderedDict()
        test_memory_buffers = collections.OrderedDict()
        if model.env_params.additional_params['prior_exp_replay']:
            tree_idxs = collections.OrderedDict()
            trans_n_tls = collections.OrderedDict()
            IS_ws = collections.OrderedDict()
        for idx, memory_queue in memory_queues.items():
            if model.env_params.additional_params['prior_exp_replay'] and model.env_params.additional_params["Policy_Type"] == "Q_Learning":
                train_memory_buffers[idx] = Memory(model.env_params.additional_params['max_buffer_size']//len(memory_queues)) 
                test_memory_buffers[idx] = []
                #test_memory_buffers[idx] = Memory((model.env_params.additional_params['max_buffer_size']//len(memory_queues))//10)
            else:
                train_memory_buffers[idx] = []
                test_memory_buffers[idx] = []
    else:
        if model.env_params.additional_params['prior_exp_replay'] and model.env_params.additional_params["Policy_Type"] == "Q_Learning":
            train_memory_buffer = Memory(model.env_params.additional_params['max_buffer_size'])
            test_memory_buffer = []  
        else:
            train_memory_buffer = []
            test_memory_buffer = []            
        
        
        
    # INITIALIZE REWARDS HOLDERS
    global_rewards = []
    if model.env_params.additional_params['save_extended_training_stats']:
        global_queues = []
        global_delays = []
        global_co2s = []
        global_trips_completed = []
    global_baseline_rewards = []
    global_greedy_rewards = []
    if model.env_params.additional_params['mode'] =='test' and tested is not None:
        tested_delays = collections.OrderedDict()
        tested_queues = collections.OrderedDict()
        tested_co2 = collections.OrderedDict()
        tested_trips_completed = collections.OrderedDict()
        tested_counters = collections.OrderedDict()
        for test in tested:
            tested_delays[test] = []
            tested_queues[test] = []
            tested_co2[test] = []
            tested_trips_completed[test] = []
            tested_counters[test] = 0
            
            
    # INITIALIZE COUNTERS 
    gr_counter = 0
    gbr_counter = 0
    ggr_counter = 0
    counter = 0
    train_counter = 0
    test_counter = 0
    max_reward = None
    reward_to_save = None
    
    # REMOVE BECAUSE 0 == GREEDY
    if greedy_reward_queue is not None:
        try:
            memory_queues.pop(0)
            reward_queues.pop(0)
        except Exception as e:
            print(e)
    # REMOVE BECAUSE 1 == BASELINE
    if baseline_reward_queue is not None:
        try:
            memory_queues.pop(1)
            reward_queues.pop(1)
        except Exception as e:
            print(e)

    elif model.env_params.additional_params['mode'] == "train":
        
        
        for idx, memory_queue in memory_queues.items():
            exp = memory_queue.recv()
            train_limit = len(exp)//10          
            if model.env_params.additional_params['separate_memory_buffers']:   
                if model.env_params.additional_params['prior_exp_replay']:
                    train_memory_buffers[idx].store(exp[train_limit:])
                    test_memory_buffers[idx].extend(exp[:train_limit])    
                    #test_memory_buffers[idx].store(exp[:train_limit])                        
                else:
                    train_memory_buffers[idx].extend(exp[train_limit:])
                    test_memory_buffers[idx].extend(exp[:train_limit])    
            else:
                if model.env_params.additional_params['prior_exp_replay']:
                    train_memory_buffer.store(exp[train_limit:])
                    test_memory_buffer.extend(exp[:train_limit])                  
                else:
                    train_memory_buffer.extend(exp[train_limit:])
                    test_memory_buffer.extend(exp[:train_limit])

    
    
    

    
    # MAIN LOOP  ( LOAD EXPERIENCE THEN TRAIN WITH MEMORY BUFFER )
    while True : 
        load_counter = 0
        if model.env_params.additional_params['mode'] == 'test':
            for idx, tested_end in tested_ends.copy().items():
                while tested_end.poll():
                    request =  tested_end.recv()
                    if type(request) == str:
                        if request == 'Done':
                            del tested_ends[idx]
                    else:
                        delay, queues, co2, trips_completed = request
                        tested_delays[tested[idx]].append(float(delay))
                        tested_queues[tested[idx]].append(float(queues))
                        tested_co2[tested[idx]].append(float(co2))
                        tested_trips_completed[tested[idx]].append(float(trips_completed))

                        writer.add_scalar("delay : network "+ str(num) + ', traffic ' + str(idx),sum(tested_delays[tested[idx]])/float(model.env_params.additional_params['n_avg_test']),tested_counters[tested[idx]]) 
                        writer.add_scalar("queue length : network "+ str(num) + ', traffic ' + str(idx),sum(tested_queues[tested[idx]])/float(model.env_params.additional_params['n_avg_test']),tested_counters[tested[idx]]) 
                        writer.add_scalar("CO2 emissions : network "+ str(num) + ', traffic ' + str(idx),sum(tested_co2[tested[idx]])/float(model.env_params.additional_params['n_avg_test']),tested_counters[tested[idx]]) 
                        writer.add_scalar("trips completed : network "+ str(num) + ', traffic ' + str(idx),sum(tested_trips_completed[tested[idx]])/float(model.env_params.additional_params['n_avg_test']),tested_counters[tested[idx]]) 

                        tested_delays[tested[idx]] = []
                        tested_queues[tested[idx]] = []
                        tested_co2[tested[idx]] = []
                        tested_trips_completed[tested[idx]] = []                   
                        tested_counters[tested[idx]] += 1




                    

        if model.env_params.additional_params['mode'] == "train":
            
            # RECEIVE EXPERIENCES FROM WORKERS AND ADD THEM TO THEIR RESPECTIVE MEMORY BUFFER(S)
            for idx, memory_queue in memory_queues.items():
                if memory_queue.poll():
                    exp = memory_queue.recv()
                    train_limit = len(exp)//10          
                    
                    if model.env_params.additional_params['separate_memory_buffers']:            
                        if model.env_params.additional_params['prior_exp_replay']:
                            train_memory_buffers[idx].store(exp[train_limit:])
                            if len(test_memory_buffers[idx]) > (model.env_params.additional_params['max_buffer_size'] // len(test_memory_buffers)):
                                test_memory_buffers[idx] = test_memory_buffers[idx][(len(test_memory_buffers[idx]) - (model.env_params.additional_params['max_buffer_size']// len(test_memory_buffers))):]
                            test_memory_buffers[idx].extend(exp[:train_limit]) 
                        else:
                            if len(train_memory_buffers[idx]) > (model.env_params.additional_params['max_buffer_size'] // len(train_memory_buffers)):
                                train_memory_buffers[idx] = train_memory_buffers[idx][(len(train_memory_buffers[idx]) - (model.env_params.additional_params['max_buffer_size']// len(train_memory_buffers))):]
                            if len(test_memory_buffers[idx]) > (model.env_params.additional_params['max_buffer_size'] // len(test_memory_buffers)):
                                test_memory_buffers[idx] = test_memory_buffers[idx][(len(test_memory_buffers[idx]) - (model.env_params.additional_params['max_buffer_size']// len(test_memory_buffers))):]
                            train_memory_buffers[idx].extend(exp[train_limit:])
                            test_memory_buffers[idx].extend(exp[:train_limit]) 
                        
                        
                    else:
                        if model.env_params.additional_params['prior_exp_replay']:
                            train_memory_buffer.store(exp[train_limit:])
                            if len(test_memory_buffer) > model.env_params.additional_params['max_buffer_size']:
                                test_memory_buffer = test_memory_buffer[(len(test_memory_buffer) - model.env_params.additional_params['max_buffer_size']):]
                            test_memory_buffer.extend(exp[:train_limit])                              
                        else:
                            if len(train_memory_buffer) > model.env_params.additional_params['max_buffer_size']:
                                train_memory_buffer = train_memory_buffer[(len(train_memory_buffer) - model.env_params.additional_params['max_buffer_size']):]
                            if len(test_memory_buffer) > model.env_params.additional_params['max_buffer_size']:
                                test_memory_buffer = test_memory_buffer[(len(test_memory_buffer) - model.env_params.additional_params['max_buffer_size']):]
                            train_memory_buffer.extend(exp[train_limit:])
                            test_memory_buffer.extend(exp[:train_limit])  


                    
            # RECEIVE REWARDS                    
            if greedy_reward_queue is not None:
                while not greedy_reward_queue.empty():
                    global_greedy_reward = greedy_reward_queue.get()
                    global_greedy_rewards.append(global_greedy_reward)   
                    
            if baseline_reward_queue is not None:
                while not baseline_reward_queue.empty():
                    global_baseline_reward = baseline_reward_queue.get()
                    global_baseline_rewards.append(global_baseline_reward)            
 
            for idx, reward_queue in reward_queues.copy().items(): 
                if reward_queue.poll():
                    request =  reward_queue.recv()
                    if type(request) == str:
                        if request == 'Done':
                            del reward_queues[idx]
                    else:
                        if model.env_params.additional_params['save_extended_training_stats']:
                            global_reward, global_delay, global_queue, global_co2, global_trip_completed,= request
                            global_rewards.append(global_reward)
                            global_queues.append(global_queue)
                            global_delays.append(global_delay)
                            global_co2s.append(global_co2)
                            global_trips_completed.append(global_trip_completed)
                        else:
                            global_reward = request
                            global_rewards.append(global_reward)

                    
                # WRITE METRICS
                if len(global_rewards) >= model.n_workers:
                    if model.env_params.additional_params['save_extended_training_stats']:
                        result_to_save = mean(global_delays[-model.n_workers:])/(model.env_params.additional_params['nb_steps_per_exp'])
                        writer.add_scalar('global delay (average of ' + str(model.n_workers) + ' experiments)', result_to_save, gr_counter)
                        global_delays = []
                        result_to_save = mean(global_queues[-model.n_workers:])/(model.env_params.additional_params['nb_steps_per_exp'])
                        writer.add_scalar('global queue (average of ' + str(model.n_workers) + ' experiments)', result_to_save, gr_counter)
                        global_queues = []
                        result_to_save = mean(global_co2s[-model.n_workers:])/(model.env_params.additional_params['nb_steps_per_exp'])
                        writer.add_scalar('global co2 (average of ' + str(model.n_workers) + ' experiments)', result_to_save, gr_counter)
                        global_co2s = []
                        result_to_save = mean(global_trips_completed[-model.n_workers:])/(model.env_params.additional_params['nb_steps_per_exp'])                        
                        writer.add_scalar('global trips completed (average of ' + str(model.n_workers) + ' experiments)', result_to_save, gr_counter)
                        global_trips_completed = []
                    reward_to_save = mean(global_rewards[-model.n_workers:])/(model.env_params.additional_params['nb_steps_per_exp'])
                    writer.add_scalar('global reward (average of ' + str(model.n_workers) + ' experiments)', reward_to_save, gr_counter)
                    global_rewards = []
                    gr_counter +=1
                    

                    if greedy_reward_queue is not None:
                        writer.add_scalar('global greedy reward', mean(global_greedy_rewards)/(model.env_params.additional_params['nb_steps_per_experience']), ggr_counter)
                        global_greedy_rewards = []
                        ggr_counter +=1


                    if baseline_reward_queue is not None:
                        writer.add_scalar('global baseline reward', mean(global_baseline_rewards)/(model.env_params.additional_params['nb_steps_per_experience']), gbr_counter)
                        global_baseline_rewards = []
                        gbr_counter +=1
                    

            if model.env_params.additional_params['target_model_update_frequency'] is not None:
                if (counter) % model.env_params.additional_params['target_model_update_frequency'] == 0:
                    update_target_graph()

            train = False if (counter) % model.env_params.additional_params['test_frequency'] == 0 else True

            

            if not train:

                if model.env_params.additional_params['separate_memory_buffers']:
                    test_mem = []
                    indexes = list(range(len(test_memory_buffers)))
                    random.shuffle(indexes)
                    while True:
                        for idx in indexes:
                            if (max(model.env_params.additional_params["batch_size"] // len(test_memory_buffers),1) + len(test_mem)) > model.env_params.additional_params["batch_size"]:
                                break
                            test_mem.extend(random.sample(test_memory_buffers[idx], max(model.env_params.additional_params["batch_size"] // len(test_memory_buffers),1)))     
                            
                        if (max(model.env_params.additional_params["batch_size"] // len(test_memory_buffers),1) + len(test_mem)) > model.env_params.additional_params["batch_size"]:
                            break
                                
                                
                else:
                    test_mem = random.sample(test_memory_buffer, model.env_params.additional_params["batch_size"])

                data_loader = DataLoader(test_mem, batch_size=model.env_params.additional_params["batch_size"], shuffle=True,
                             collate_fn=collate, drop_last=True)            
                model.eval()
                epochs = 1


            else:
                
                if model.env_params.additional_params['separate_memory_buffers']:
                    train_mem = []
                    indexes = list(range(len(train_memory_buffers)))
                    random.shuffle(indexes)
                    buffer_indexes = []
                    while True:
                        for idx in indexes:
                            if (max(model.env_params.additional_params["batch_size"] // len(train_memory_buffers),1) + len(train_mem)) > model.env_params.additional_params["batch_size"]:
                                break
                            if model.env_params.additional_params['prior_exp_replay']:
                                tree_idx, train_mem_, IS_w = train_memory_buffers[idx].sample(max(model.env_params.additional_params["batch_size"] // len(train_memory_buffers),1))

                                #print("train_mem_", len(train_mem_[0]), train_mem_[0])

                                trans_n_tls[idx] = []
                                IS_ws[idx] = []
                                tree_idxs[idx] = tree_idx
                                for trans, w in zip(train_mem_, IS_w):
                                    trans_n_tls[idx].append(len(trans[0][1]))
                                    IS_ws[idx].append(torch.from_numpy(w))

                                train_mem.extend(train_mem_[0])
                                buffer_indexes.append(idx)

                            else:
                                train_mem.extend(random.sample(train_memory_buffers[idx], max(model.env_params.additional_params["batch_size"] // len(train_memory_buffers),1)))      
                            
                        if (max(model.env_params.additional_params["batch_size"] // len(train_memory_buffers),1) + len(train_mem)) > model.env_params.additional_params["batch_size"]:
                            break

                        
                else:
                    if model.env_params.additional_params['prior_exp_replay']: 
                        tree_idx, train_mem_, IS_w = train_memory_buffer.sample(model.env_params.additional_params["batch_size"])
                        train_mem = [tm[0] for tm in train_mem_]
                        trans_n_tls = []
                        IS_ws = []
                        for trans, w in zip(train_mem_, IS_w):
                            trans_n_tls.append(len(trans[0][1]))
                            IS_ws.append(torch.from_numpy(w))

                    else:
                        train_mem = random.sample(train_memory_buffer, model.env_params.additional_params["batch_size"])


                    
                    
                #print("train_mem", train_mem)
                data_loader = DataLoader(train_mem, batch_size=model.env_params.additional_params["batch_size"], shuffle=False if model.env_params.additional_params['prior_exp_replay'] else True,
                             collate_fn=collate, drop_last=True)
                model.train()
                epochs = 1


            # TRAIN ON MEMORY BUFFERS

            for idx, (g_state, imm_rewards, actions, choices, forced_mask, forced_value, actions_sizes, future_rewards, next_g_state, prev_imm_rewards, next_choices, next_forced_mask, next_forced_value, next_actions_sizes) in enumerate(data_loader):  # choices
                if idx >= epochs:
                    break

                # FOR Q-LEARNING 

                if model.env_params.additional_params["Policy_Type"] == "Q_Learning" :
                # FORWARD
                    # MODEL
                    if model.env_params.additional_params['target_model_update_frequency'] is None or  model.env_params.additional_params['double_DQN']:
                        
                        if model.env_params.additional_params["GCN"]:
                            full_batched_graph = dgl.batch([g_state, next_g_state])  
                        elif not model.env_params.additional_params["GCN"]:
                            full_batched_graph = g_state + next_g_state
                        #print("fbg", full_batched_graph)
                        hid, Q = model.forward(full_batched_graph, device, learning = True, joint = True, actions_sizes = torch.cat((actions_sizes, next_actions_sizes), 0))   
                        if model.env_params.additional_params['value_model_based']:
                            model_hid, model_NEXT_hid = hid.chunk(2, dim = 0)   
                        else:
                            model_Q, model_NEXT_Q = (Q.chunk(2, dim = 0)) if model.env_params.additional_params['policy'] == 'binary' else split_list(Q)
                    else:
                        model_hid, model_Q = model.forward(g_state, device, learning = True, actions_sizes = actions_sizes)    

                    with torch.no_grad():
                        # TARGET
                        if model.env_params.additional_params['target_model_update_frequency'] is not None:
                            target_NEXT_hid, target_NEXT_Q = target_model.forward(next_g_state, device, learning = True, actions_sizes = next_actions_sizes, testing = True)   

                            
                            
                            
                            
                    if model.env_params.additional_params['policy'] != 'binary':
                        next_maxs = [0]*len(actions_sizes)
                        outputs = [0]*len(next_actions_sizes)
                        for dim in list(set(next_actions_sizes)):
                            positions = [i for i, n in enumerate(next_actions_sizes) if n == dim]
                            m_q =  torch.cat(tuple(model_Q[i].view(1,-1) for i in positions),dim=0)
                            acts =  torch.cat(tuple(actions[i].view(1,-1) for i in positions),dim=0)
                            outp = m_q.gather(1, acts.to(device))
                            with torch.no_grad():
                                if (model.env_params.additional_params['target_model_update_frequency'] is not None and not model.env_params.additional_params['double_DQN']):
                                    _, n_m_l = torch.max(torch.cat(tuple(target_NEXT_Q[i].view(1,-1) for i in positions),dim=0), dim = 1)                             
                                else:
                                    _, n_m_l = torch.max(torch.cat(tuple(model_NEXT_Q[i].view(1,-1) for i in positions),dim=0), dim = 1)   
                                    
                                if (model.env_params.additional_params['target_model_update_frequency'] is None or model.env_params.additional_params['double_DQN']):
                                    n_q = torch.cat(tuple(model_NEXT_Q[i].view(-1,dim) for i in positions),dim=0)
                                else:
                                    n_q = torch.cat(tuple(target_NEXT_Q[i].view(-1,dim) for i in positions),dim=0)

                                if model.env_params.additional_params['correct_actions']:
                                    n_f_m = next_forced_mask[positions].view(n_m_l.size()).to(device)
                                    n_f_v = next_forced_value[positions].view(n_m_l.size()).to(device)
                                    n_m_l = n_m_l.masked_scatter(n_f_m.eq(1),n_f_v.masked_select(n_f_m.eq(1)))   
                                n_q = n_q.to(device)
                                n_m_l = n_m_l.to(device)
                                n_m = n_q.gather(1, n_m_l.view(-1,1))  
                            for idx,position in enumerate(positions):
                                with torch.no_grad():
                                    next_maxs[position] = n_m[idx]
                                outputs[position] = outp[idx]                                        

                        outputs = torch.cat(tuple(outputs),0)
                        next_maxs = torch.cat(tuple(next_maxs),0)                         

                    else:
                        with torch.no_grad():

                            # GET BEST ACTIONS
                            _, next_maxs_locs = torch.max(target_NEXT_Q if (model.env_params.additional_params['target_model_update_frequency'] is not None and not model.env_params.additional_params['double_DQN']) else model_NEXT_Q, dim = 1)

                            NEXT_Q = model_NEXT_Q if (model.env_params.additional_params['target_model_update_frequency'] is None or model.env_params.additional_params['double_DQN']) else target_NEXT_Q

                            # OPTIONNAL : CORRECTION
                            if model.env_params.additional_params['correct_actions']:
                                next_forced_mask = next_forced_mask.to(device)
                                next_forced_value = next_forced_value.to(device)
                                next_maxs_locs = next_maxs_locs.masked_scatter(next_forced_mask.eq(1),next_forced_value.masked_select(next_forced_mask.eq(1)))

                            # GET BEST ACTIONS EVALUATIONS
                            next_maxs = NEXT_Q.gather(1, next_maxs_locs.view(-1,1))   
                        outputs = model_Q.gather(1, actions.to(device).unsqueeze(1)).squeeze()                        
                        
                        
                        
                    with torch.no_grad():
                        targets = Variable(imm_rewards.to(device) + model.env_params.additional_params["time_gamma"] * next_maxs.squeeze(), requires_grad = False)
                
                if model.env_params.additional_params['train_on_real_choices_only']:

                    loss = F.mse_loss(outputs.masked_select(choices.to(device)), targets.to(device).detach().masked_select(choices.to(device)))          
                    if model.env_params.additional_params['value_model_based']:
                        value_model_pred = model.value_transition_model[0].forward(torch.cat((hid, actions),1))
                        value_model_pred = value_model_pred + hid 
                        loss_v = F.mse_loss(value_model_pred, NEXT_hid.detach())                        
                else:

                    if model.env_params.additional_params['prior_exp_replay'] and train:
                        buffer_idx_counter = 0
                        loss = F.mse_loss(outputs, targets.to(device).detach(), reduction = 'none').squeeze()
                        #print("loss", loss)
                        if model.env_params.additional_params['separate_memory_buffers']:
                            for idx in buffer_indexes:
                                for l, w in zip(trans_n_tls[idx], IS_ws[idx]):
                                    #print("l :", l , "w :", w)
                                    loss[buffer_idx_counter:buffer_idx_counter + l] *= w.squeeze()
                                    buffer_idx_counter += l
                        else:
                            for l, w in zip(trans_n_tls, IS_ws):
                                loss[buffer_idx_counter:buffer_idx_counter + l] *= w.squeeze()
                                buffer_idx_counter += l

                        loss = loss.mean()
                    else:   
                        if model.env_params.additional_params['criterion'] == 'smooth_l1_loss':                         
                            loss = F.smooth_l1_loss(outputs, targets.to(device).detach())
                        else : 
                            loss = F.mse_loss(outputs, targets.to(device).detach())

                            
                    if model.env_params.additional_params['value_model_based']:
                        value_model_pred = model.value_transition_model[0].forward(torch.cat((hid, actions.type(torch.FloatTensor).to(device).unsqueeze(1)),1))
                        value_model_pred = value_model_pred + hid 
                        loss_v = F.mse_loss(value_model_pred, NEXT_hid.detach())

                # NAIVE PREDICTION AGAINST WHICH  BENCHMARK
                with torch.no_grad():
                    naive_preds = prev_imm_rewards.to(device) * (1 + 1/(((1/model.env_params.additional_params['time_gamma']) - 1))) # PRESENT VALUE OF A PERPETUITY 
                    
                    naive_loss = F.mse_loss(naive_preds.to(device), targets.to(device).detach())           
                    performance_ratio = 1 - (loss / naive_loss)


                if not train:
                    writer.add_scalar('general_test_loss', loss.item(), train_counter)   
                    if model.env_params.additional_params['value_model_based']:
                        writer.add_scalar('general_test_value_model_loss', loss_v, train_counter)
                    writer.add_scalar('general_test_performance_ratio', performance_ratio, train_counter)

                else :
                    if model.env_params.additional_params['prior_exp_replay']:
                        with torch.no_grad():
                            buffer_idx_counter = 0
                            #print("buffer_indexes", buffer_indexes, "trans_n_tls", trans_n_tls)
                            if model.env_params.additional_params['separate_memory_buffers']:
                                for idx in buffer_indexes:
                                    abs_errors = []
                                    for l in trans_n_tls[idx]:
                                        abs_error = torch.nn.functional.l1_loss(outputs[buffer_idx_counter:buffer_idx_counter + l], targets.to(device).detach()[buffer_idx_counter:buffer_idx_counter + l]).item()
                                        buffer_idx_counter += l
                                        abs_errors.append(abs_error/l)
                                    train_memory_buffers[idx].batch_update(tree_idxs[idx],np.asarray(abs_errors))

                            else:
                                abs_errors = []
                                for l in trans_n_tls:
                                    abs_error = torch.nn.functional.l1_loss(outputs[buffer_idx_counter:buffer_idx_counter + l], targets.to(device).detach()[buffer_idx_counter:buffer_idx_counter + l]).item()
                                    buffer_idx_counter += l
                                    abs_errors.append(abs_error/l)                                          
                                train_memory_buffer.batch_update(tree_idx, np.asarray(abs_errors))

                            
                    writer.add_scalar('general_train_loss', loss.item(), train_counter)                        
                    writer.add_scalar('general_train_performance_ratio', performance_ratio, train_counter)
                    loss /= model.env_params.additional_params['accumulation_steps']
                    if model.env_params.additional_params['value_model_based']:
                        writer.add_scalar('general_train_value_model_loss', loss_v.item(), train_counter)
                        loss_v /= model.env_params.additional_params['accumulation_steps']
                        loss = 1*loss + 1*loss_v
                      

                    loss.backward()

                        
                    if (train_counter+1) % model.env_params.additional_params['accumulation_steps'] == 0:

                        model.optimizer.step()
                        model.optimizer.zero_grad()
                        

                if model.env_params.additional_params["tb_embed_viz"] and (idx+1) % model.env_params.additional_params["tb_embed_viz_frequency"] == 0 :
                    writer.add_embedding(g.ndata['hid'], metadata=list(g.ndata['node_type']), label_img=None, global_step= idx, tag='node_embedding', metadata_header=None)


                if not train:
                    test_counter +=1
                else:
                    train_counter +=1    


            counter +=1


            # SAVE MODEL PARAMETERS ONLY IF PERFORMANCE IS INCREASING AND 
            if max_reward is None and reward_to_save is not None:
                max_reward = reward_to_save
                
            if max_reward is not None and reward_to_save is not None:
                if reward_to_save >= max_reward:
                    max_reward = reward_to_save
                    #torch.save(model, model.env_params.additional_params['trained_model_path'] + "/model.pt")
                    torch.save(model.state_dict(), model.env_params.additional_params['save_model_path'] + "/params.pt")

            if counter% model.env_params.additional_params['save_params_frequency']==0:
                torch.save(model.state_dict(), model.env_params.additional_params['save_model_path'] + "/params_checkpoint.pt")

            while not comput_model_queue.empty():
                try:
                    _ = comput_model_queue.get_nowait()
                except:
                    pass

            comput_model_queue.put_nowait(model)

        
        
        
        else:
            counter+=1
            
        if (counter) % model.env_params.additional_params["clear_after_n_epoch"] == 0:  
            clear_output(wait=True)   