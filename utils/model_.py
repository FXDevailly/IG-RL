import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch
import numpy as np 
import dgl
from functools import partial
from torch.autograd import Variable
import copy
from utils.mdn import *
from utils.graph_tools import *
from IPython.display import clear_output
import glob
import atexit
import time
import traceback

  
    
class Convolutional_Message_Passing_Framework(nn.Module):
    def __init__(self, policy, noisy, num_attention_heads, bias_before_aggregation , nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, state_first_dim_only, gaussian_mixture, n_gaussians, value_model_based, use_attention,  separate_actor_critic, n_actions, rl_learner_type, std_attention, state_vars, n_convolutional_layers, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = 1, rel_num_bases = -1, norm = True, bias = True, activation = F.relu, use_message_module = True, use_aggregation_module = False, dropout = False, resnet = False, dueling = False):
        
        
        
        
        """
        num_attention_heads : number of attention heads to be used 
        bias_before_aggregation : if True, a different bias is used for every type of edge 
        nonlinearity_before_aggregation : if True, nonlinearity is applied to every message before aggregation 
        share_initial_params_between_actions : if True, the parameters of the RL components are shared across actions
        multidimentional_attention : if True, attention is applied per dimension of messages (Hadammard product)
        state_first_dim_only : if True, the state is only present in the initial representation. if False, it is concatenated to every embedding and used "as is" for every propagation. 
        gaussian_mixture : if True, a MDN is used 
        n_gaussians : Numbers of gaussians to use in MDN 
        value_model_based : if True, a model_based RL approach is used 
        use_prediction_module : if False, embeddings obtained after a given propagation are used as predicion values (ie, they are not transformed by additional layers)
        use_attention : if True, attention element-wise attention mechanisms are used to weight messages
        separate_actor_critic : if True, two separate architectures (one for the actor and one for the critic) perform propagations, with their own sets of parameters
        n_pred : number of values to predict 
        learner_type : if "actor" or "critic" or "actor-critic" specific behaviour will be specified in the fields between ############ Start and ############### End
        std_attention : if True, attention coefficients corresponding to messages of the same type leading to the same node will be normalized to sum up to one
        state_vars : list containing the names of variables representing the state (RL)
        n_convolutional_layers : number of layers with different parameters we want to use.
        num_nodes_types : number of different types of nodes which exist in the studied graphs 
        nodes_types_num_bases : if < 0 no basis decomposition is used for the nodes parameters (Gating/GRU). If > 0 there will be as many bases for node parameters as indicated (up to a maximum of 'num_nodes_types')
        node_state_dim : the size of the complete node representation (state_size + embedding_size)
        node_embedding_dim : the size of the node embedding (computed representation)
        num_rels : number of different types of edges which exist in the studied graphs
        rel_num_bases : if < 0 no basis decomposition is used for the edges parameters (message/request/attention). If > 0 there will be as many bases for edges parameters as indicated (up to a maximum of 'num_rels')
        n_hidden_message : number of hidden layers of neural networks used in the message module (to compute :message, request, attention) 
        n_hidden_aggregation : number of hidden GRU layers used to update node embeddings 
        n_hidden_prediction : number of hidden layers used to make a prediction in the prediction module from the embeddings of the nodes 
        hidden_layers_size : deprecated
        prediction_size : deprecated
        num_propagations : default number of propagations which are performed ON EVERY DIFFERENT LAYER during a forward pass if no 'num_propagations' argument is passed to the forward function.
        norm : if True, messages of the same type leading to the same node will be averaged 
        bias : if True, a bias will be included with every layer (message/request/attention/GRU/prediction...)
        activation: The activation function which will be used to add non linearity in the architecture
        use_message_module : deprecated
        use_aggregation_module : if False, the embedding is simply the aggregation of the received messages. if True, a GRU/Gate is used to compute the new embedding using the aggregation of the received messages and the previous embedding.
        dropout : if True, dropout will be used with the specified probability ( not yet properly implemented )
        resnet : if True, a residual connection is added between the previous embedding of a node and its new embedding (ie. the aggregation module only has to learn to output the relevant difference between the previous and new embedding)


        """        
        super(Convolutional_Message_Passing_Framework, self).__init__() 
        self.policy = policy
        self.noisy = noisy
        self.num_attention_heads = num_attention_heads
        self.bias_before_aggregation = bias_before_aggregation
        self.nonlinearity_before_aggregation = nonlinearity_before_aggregation
        self.share_initial_params_between_actions = share_initial_params_between_actions
        self.multidimensional_attention = multidimensional_attention
        self.state_first_dim_only = state_first_dim_only, 
        self.gaussian_mixture = gaussian_mixture
        self.n_gaussians = n_gaussians
        self.value_model_based = value_model_based
        self.use_attention = use_attention
        self.separate_actor_critic = separate_actor_critic
        self.n_actions = n_actions
        self.rl_learner_type = rl_learner_type
        self.std_attention = std_attention
        self.state_vars = state_vars 
        self.state_vars = state_vars
        self.n_convolutional_layers = n_convolutional_layers
        self.num_nodes_types = num_nodes_types 
        self.nodes_types_num_bases = nodes_types_num_bases
        self.node_state_dim = node_state_dim
        self.node_embedding_dim = node_embedding_dim
        self.num_rels = num_rels
        self.n_hidden_message = n_hidden_message 
        self.n_hidden_aggregation = n_hidden_aggregation 
        self.n_hidden_prediction = n_hidden_prediction
        self.hidden_layers_size = hidden_layers_size
        self.n_hidden_aggregation = n_hidden_aggregation
        self.prediction_size = prediction_size
        self.num_propagations = num_propagations
        self.rel_num_bases = rel_num_bases
        self.norm = norm 
        self.bias = bias 
        self.activation = activation
        self.use_message_module = use_message_module
        self.use_aggregation_module = use_aggregation_module
        self.dropout = dropout
        self.resnet = resnet 
        self.dueling = dueling if self.rl_learner_type =='Q_Learning' else False
        
        
        self.conv_layers = nn.ModuleList()
        
        if self.dueling:
            self.rl_learner_type = 'actor_critic'
        if self.value_model_based:
            self.value_transition_model = nn.ModuleList()
        
        if 'actor' in self.rl_learner_type.lower() and 'critic' in self.rl_learner_type.lower() and self.separate_actor_critic :
            self.critic_conv_layers = nn.ModuleList()
            self.actor_conv_layers = nn.ModuleList()
        
            for i in range(self.n_convolutional_layers):
                
                
                #if i < (self.n_convolutional_layers - 1):
                critic_conv = Relational_Message_Passing_Framework(policy, noisy, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers,True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, bias = bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = False, dropout = dropout, resnet = resnet)
                actor_conv = Relational_Message_Passing_Framework(policy, noisy, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers,True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, bias = bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = False, dropout = dropout, resnet = resnet)
                #else:
                    #critic_conv = Relational_Message_Passing_Framework(policy, noisy, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers, True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, n_actions, 'critic', std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, bias = bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = True, dropout = dropout, resnet = resnet)
                    #actor_conv = Relational_Message_Passing_Framework(policy, noisy, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers, True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, n_actions, 'actor', std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, bias = bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = False, dropout = dropout, resnet = resnet)

                self.critic_conv_layers.append(critic_conv)
                self.actor_conv_layers.append(actor_conv)
                
    
    
        else: 
            for i in range(self.n_convolutional_layers):
                #if i < (self.n_convolutional_layers - 1):
                conv = Relational_Message_Passing_Framework(policy, noisy, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers, True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, n_actions, self.rl_learner_type, std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, bias = bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = False, dropout = dropout, resnet = resnet)
                #else:
                    #conv = Relational_Message_Passing_Framework(policy, noisy, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, self.state_first_dim_only, self.n_convolutional_layers, True if i == 0 else False, gaussian_mixture, n_gaussians, value_model_based, use_attention, n_actions, self.rl_learner_type, std_attention, self.state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, bias = bias, activation = F.relu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, is_final_convolutional_layer = True, dropout = dropout, resnet = resnet)

                self.conv_layers.append(conv)
                
                
        #if (self.policy != 'binary' and (not ('actor' in self.rl_learner_type.lower() and 'critic' in self.rl_learner_type.lower())) :
        prediction_module_num_nodes_types = num_nodes_types
        #else :
            #prediction_module_num_nodes_types = 1
                
        self.prediction_module = Prediction_Module(self.dueling, policy, noisy, share_initial_params_between_actions, gaussian_mixture, n_gaussians, value_model_based, n_actions, self.rl_learner_type, use_message_module, prediction_module_num_nodes_types, self.nodes_types_num_bases, n_hidden_prediction, self.hidden_layers_size, in_feat = self.node_embedding_dim, out_feat = self.prediction_size, bias = bias, activation = F.relu, norm = False, dropout = dropout)
                
                
    def init_hidden(self, graph, device):
        # INITIALIZATION OF NODE HIDDEN AND MEMORY HIDDEN

        graph.ndata.update({"hid":torch.zeros(graph.number_of_nodes(), self.node_embedding_dim, dtype = torch.float32)}) # requires_grad = True
        graph.ndata["hid"].to(device)  
        if self.n_hidden_aggregation > 0 :
            graph.ndata.update({"memory_input":torch.zeros(graph.number_of_nodes(), self.node_embedding_dim, dtype = torch.float32)}) # requires_grad = True
            graph.ndata["memory_input"].to(device)  
            for idx in range(self.n_hidden_aggregation-1):
                graph.ndata.update({str('memory_' + str(idx)): torch.zeros(graph.number_of_nodes(),self.node_embedding_dim, dtype = torch.float32)})             
                graph.ndata[str('memory_' + str(idx))].to(device)

            graph.ndata.update({"memory_output":torch.zeros(graph.number_of_nodes(), self.node_embedding_dim, dtype = torch.float32)}) # requires_grad = True
            graph.ndata["memory_output"].to(device)
        
        self.save_hidden(graph)
        return graph
    
    def save_hidden(self, graph, role = None):
        if role == 'critic':
            self.critic_hid = graph.ndata['hid']
            if self.n_hidden_aggregation > 0 :
                self.critic_memory_input = graph.ndata["memory_input"]
                self.critic_memory_hid = []
                for idx in range(self.n_hidden_aggregation-1):
                    self.critic_memory_hid.append(graph.ndata[str('memory_' + str(idx))])
                self.critic_memory_output = graph.ndata["memory_output"]
        elif role == 'actor':
            self.actor_hid = graph.ndata['hid']
            if self.n_hidden_aggregation > 0 :
                self.actor_memory_input = graph.ndata["memory_input"]
                self.actor_memory_hid = []
                for idx in range(self.n_hidden_aggregation-1):
                    self.actor_memory_hid.append(graph.ndata[str('memory_' + str(idx))])
                self.actor_memory_output = graph.ndata["memory_output"]            
        else:
            self.hid = graph.ndata['hid']
            if self.n_hidden_aggregation > 0 :
                self.memory_input = graph.ndata["memory_input"]
                self.memory_hid = []
                for idx in range(self.n_hidden_aggregation-1):
                    self.memory_hid.append(graph.ndata[str('memory_' + str(idx))])
                self.memory_output = graph.ndata["memory_output"]
            
    def get_hidden(self, graph, role = None):
        if role == 'critic':
            # INITIALIZATION OF NODE HIDDEN AND MEMORY HIDDEN
            graph.ndata["hid"] = self.critic_hid
            if self.n_hidden_aggregation > 0 :
                graph.ndata["memory_input"] = self.critic_memory_input
                for idx in range(self.n_hidden_aggregation-1):
                    graph.ndata[str('memory_' + str(idx))] = self.critic_memory_hid[idx]
                graph.ndata["memory_output"] = self.critic_memory_output          
        elif role == 'actor':
            # INITIALIZATION OF NODE HIDDEN AND MEMORY HIDDEN
            graph.ndata["hid"] = self.actor_hid
            if self.n_hidden_aggregation > 0 :
                graph.ndata["memory_input"] = self.actor_memory_input
                for idx in range(self.n_hidden_aggregation-1):
                    graph.ndata[str('memory_' + str(idx))] = self.actor_memory_hid[idx]
                graph.ndata["memory_output"] = self.actor_memory_output          
        else:
            # INITIALIZATION OF NODE HIDDEN AND MEMORY HIDDEN
            graph.ndata["hid"] = self.hid
            if self.n_hidden_aggregation > 0 :
                graph.ndata["memory_input"] = self.memory_input
                for idx in range(self.n_hidden_aggregation-1):
                    graph.ndata[str('memory_' + str(idx))] = self.memory_hid[idx]
                graph.ndata["memory_output"] = self.memory_output
        return graph
            
        
    def forward(self, graph, device, learning = False, joint = False, testing = False, actions_sizes = None):
        # SEND TENSORS TO DEVICE 
        graph.ndata['state'].to(device)
        graph.edata['rel_type'].to(device)
        if self.norm:
            graph.edata['norm'].to(device)
        graph.ndata["node_type"].to(device)
        
        
        
        
        #print("rel type", graph.edata['rel_type'])
        #sys.exit()
        
        actions_sizes_list = tuple(actions_sizes.tolist())
        #print('actions_sizes_list', actions_size_list)

        """
        # INITIALIZATION OF NODE HIDDEN AND MEMORY HIDDEN

        graph.ndata.update({"hid":torch.zeros(graph.number_of_nodes(), self.node_embedding_dim, dtype = torch.float32)}) # requires_grad = True
        graph.ndata["hid"].to(device)

        graph.ndata.update({"memory_input":torch.zeros(graph.number_of_nodes(), self.node_embedding_dim, dtype = torch.float32)}) # requires_grad = True
        graph.ndata["memory_input"].to(device)    


        if self.n_hidden_aggregation > 0 :

            for idx in range(self.n_hidden_aggregation-1):
                graph.ndata.update({str('memory_' + str(idx)): torch.zeros(graph.number_of_nodes(),self.node_embedding_dim, dtype = torch.float32)})             
                graph.ndata[str('memory_' + str(idx))].to(device)

            graph.ndata.update({"memory_output":torch.zeros(graph.number_of_nodes(), self.node_embedding_dim, dtype = torch.float32)}) # requires_grad = True
            graph.ndata["memory_output"].to(device)
        """

        if self.policy != 'binary':
            #print('self rl learner', self.rl_learner_type.lower())
            if not ('actor' in self.rl_learner_type.lower() and 'critic' in self.rl_learner_type.lower()):
                #print('AAAAAAAA')
                identifier = [self.nodes_types['phase']]
            else:
                #print("BBBBBBBB")
                identifier = [self.nodes_types['tl'],self.nodes_types['phase']]
        else:
            #print("a")
            identifier = [self.nodes_types['tl']]

        
        if 'actor' in self.rl_learner_type.lower() and 'critic' in self.rl_learner_type.lower() :
            if self.separate_actor_critic :  
                critic_graph = copy.deepcopy(graph)
                actor_graph = copy.deepcopy(graph)
                if not self.dueling:
                    critic_graph = self.get_hidden(critic_graph, 'critic')
                    critic_graph = self.get_hidden(actor_graph, 'actor')
                else:
                    critic_graph = self.init_hidden(critic_graph)
                    actor_graph = self.init_hidden(actor_graph)
                for critic_layer, actor_layer in zip(self.critic_conv_layers, self.actor_conv_layers):
                    critic_layer.forward(critic_graph, device, testing)            
                    actor_layer.forward(actor_graph, device, testing)  
                if not self.dueling:                    
                    self.save_hidden(critic_graph, 'critic')
                    self.save_hidden(actor_graph, 'actor')
                    
                    critic_graph.ndata['hid'] = critic_graph.ndata['hid'].masked_scatter(actor_graph.ndata['node_type'].eq(self.nodes_types['connection']), actor_graph.ndata['hid'].masked_select(actor_graph.ndata['node_type'].eq(self.nodes_types['connection'])))
                    
                new_filt = partial(filt, identifier = identifier)
                subgraph = critic_graph.subgraph(list(critic_graph.filter_nodes(new_filt)))
                subgraph.copy_from_parent()

                if self.policy != 'binary':
                    nt, hid, v = self.prediction_module.predict(subgraph,device, testing)             
                else:
                    hid, a, v = self.prediction_module.predict(subgraph,device, testing)  
                    
            else:
                if not self.dueling:
                    graph = self.get_hidden(graph, device)
                else:
                    #print("yolo")
                    graph = self.init_hidden(graph, device)
                    
                for layer in self.conv_layers:
                    #print("graph", graph.ndata)
                    layer.forward(graph, device, testing)
                    #print("YOLO")
                    
                if not self.dueling:
                    self.save_hidden(graph)
                    #print("self.hid", self.hid)
                new_filt = partial(filt, identifier = identifier)
                subgraph = graph.subgraph(list(graph.filter_nodes(new_filt)))
                subgraph.copy_from_parent()   
                #print("subgraph", subgraph.ndata)
                #print("graph node types", subgraph.ndata['node_type'])
                
                if self.policy != 'binary':
                    nt, hid, v = self.prediction_module.predict(subgraph, device, testing)    
                else:
                    #print("b")
                    #hid, v =  self.prediction_module.predict(subgraph, device)    
                    #print("hid", hid.size(), "v", v.size())
                    hid, a, v = self.prediction_module.predict(subgraph, device, testing)                      
                    
                    
            if self.policy != 'binary':
                #print("graph node types", subgraph.ndata['node_type'])
                nt = nt.to(device)
                a = v.masked_select(nt.eq(self.nodes_types['connection']))
                v = v.masked_select(nt.eq(self.nodes_types['tl']))
                #print('a', a.size())
                #print('v', v.size())
                #print('actions_sizes_list',actions_sizes_list)
                #print('l_a', l_a)
                l_q = list(a.split(actions_sizes_list))
                a = l_q
                
                #print('actions_sizes_list',actions_sizes_list)
                #print('l_a', l_a)
            if self.dueling:
                if self.policy != 'binary':
                    if not learning:
                        l_Q = torch.zeros(actions_sizes.size(), dtype=torch.int8, device=device)                       
                    else:
                        l_Q = [0]*len(l_q)
                    for dim in list(set(actions_sizes_list)):
                        positions = [i for i, n in enumerate(actions_sizes_list) if n == dim]
                        a_ = torch.cat(tuple(l_q[i].view(1,-1) for i in positions),dim=0)
                        v_ = v[positions]
                        q_ = (v_.view(-1,1) + (a_ - torch.mean(a_, dim = 1).unsqueeze(1))).squeeze()
                        q_ = q_.view(-1,dim)
                        for idx,position in enumerate(positions):
                            if not learning:
                                _ , l_Q[position] = torch.max(q_[idx],dim = 0)             
                            else:
                                l_Q[position] = q_[idx]  
                        
                    #print('l_Q',l_Q)
                    return hid, l_Q
                else:
                    #print("a", a.size(), "v", v.size())
                    return hid, (v.unsqueeze(1) + (a - torch.mean(a, dim = 1).unsqueeze(1))).squeeze()
                
            else:
                return None, a, v
            
        else:
            graph = self.init_hidden(graph, device)
            for layer in self.conv_layers:
                layer.forward(graph, device, testing)

            #print('identifier', identifier)
            new_filt = partial(filt, identifier = identifier)
            subgraph = graph.subgraph(list(graph.filter_nodes(new_filt)))
            subgraph.copy_from_parent()    
            if self.policy != 'binary':            
                nt, hid, Q = self.prediction_module.predict(subgraph, device, testing)
                
                #print('Q size', Q.size())
                #print('len actions sizes', np.sum(np.array(actions_sizes_list)))
                l_q = list(Q.split(actions_sizes_list))
                if not learning:
                    l_Q = torch.zeros(actions_sizes.size(), dtype=torch.int8, device=device)                       
                else:
                    l_Q = [0]*len(l_q)

                for dim in list(set(actions_sizes_list)):
                    positions = [i for i, n in enumerate(actions_sizes_list) if n == dim]
                    q_ = torch.cat(tuple(l_q[i].view(1,-1) for i in positions),dim=0)
                    q_ = q_.view(-1,dim)
                    for idx,position in enumerate(positions):
                        if not learning:
                            _ , l_Q[position] = torch.max(q_[idx],dim = 0)             
                        else:
                            l_Q[position] = q_[idx]  

                #print('l_Q',l_Q)
                return hid, l_Q
            else:
                hid, Q = self.prediction_module.predict(subgraph, device, testing)
                return hid, Q 

            
        
            
    
    
    
    
    
    
class Relational_Message_Passing_Framework(nn.Module):
    def __init__(self, policy, noisy, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, state_first_dim_only, n_convolutional_layers, is_first_layer, gaussian_mixture, n_gaussians, value_model_based, use_attention, n_actions, rl_learner_type, std_attention, state_vars, num_nodes_types, nodes_types_num_bases, node_state_dim, node_embedding_dim, num_rels, n_hidden_message, n_hidden_aggregation, n_hidden_prediction, hidden_layers_size, prediction_size, num_propagations = 1, rel_num_bases = -1, norm = True, bias = True, activation = F.relu, use_message_module = True, use_aggregation_module = False, is_final_convolutional_layer = False, dropout = False, resnet = False):
        
        
        super(Relational_Message_Passing_Framework, self).__init__()
        self.policy = policy
        self.noisy = noisy
        self.is_first_layer = is_first_layer
        self.state_first_dim_only = state_first_dim_only
        self.n_convolutional_layers = n_convolutional_layers
        self.rl_learner_type = rl_learner_type
        self.state_vars = state_vars
        self.is_final_convolutional_layer = is_final_convolutional_layer
        self.prediction_size = prediction_size
        self.num_nodes_types = num_nodes_types
        self.nodes_types_num_bases = nodes_types_num_bases
        self.use_aggregation_module = use_aggregation_module
        self.node_state_dim = node_state_dim
        self.node_embedding_dim = node_embedding_dim
        self.norm = norm
        self.hidden_layers_size = hidden_layers_size
        self.num_propagations = num_propagations 
        self.layers = nn.ModuleDict()
        self.num_rels = num_rels
        self.rel_num_bases = rel_num_bases
        self.bias = bias
        self.activation = activation
        
        data = []
        
        if is_first_layer:
            data.append('state')
            in_feat = self.node_state_dim
            if n_convolutional_layers == 1 or not state_first_dim_only:
                in_feat += self.node_embedding_dim
                data.append('hid')
        else:
            if n_convolutional_layers == 1 or not state_first_dim_only:
                in_feat += self.node_state_dim
                data.append('state')
            in_feat = self.node_embedding_dim
            data.append('hid')

            
        
        self.layers['message_module'] = Message_Module(noisy, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, multidimensional_attention, data, use_attention, std_attention, resnet, use_message_module, self.state_vars, self.num_rels, self.rel_num_bases, n_hidden_message, self.hidden_layers_size, in_feat = in_feat , out_feat = self.node_embedding_dim, bias = bias, activation = F.relu, norm = norm, dropout = dropout)

        self.layers['aggregation_module'] = Aggregation_Module(noisy, use_attention, num_attention_heads, multidimensional_attention, resnet, use_aggregation_module, self.num_nodes_types, self.nodes_types_num_bases, n_hidden_aggregation, self.hidden_layers_size, in_feat = self.node_embedding_dim, out_feat = self.node_embedding_dim, bias = bias, activation = F.relu, norm = False, dropout = dropout)
        
        #if self.is_final_convolutional_layer:
            #self.layers['prediction_module'] = Prediction_Module(policy, noisy, share_initial_params_between_actions, gaussian_mixture, n_gaussians, value_model_based, n_actions, rl_learner_type, use_message_module, self.num_nodes_types, self.nodes_types_num_bases, n_hidden_prediction, self.hidden_layers_size, in_feat = self.node_embedding_dim, out_feat = self.prediction_size, bias = bias, activation = F.relu, norm = False, dropout = dropout)


        
    def forward(self, graph, device, testing = False, num_propagations = None):
        if not num_propagations :
            num_propagations = self.num_propagations

        for _ in range(num_propagations):
            self.layers['message_module'].propagate(graph, device, testing)
            self.layers['aggregation_module'].aggregate(graph, device, testing)
                
                
                
        # AUTRE FONCTION POUR CLASSIFY ? 
        #if self.is_final_convolutional_layer:
            """if self.policy != 'binary':
                if not ('actor' in self.rl_learner_type.lower() and 'critic' in self.rl_learner_type.lower()):
                    identifier = [5]
                else:
                    identifier = [1,5]
            else:
                identifier = [1]
            new_filt = partial(filt, identifier = identifier)
            #print("new_filt",new_filt())
            #sys.exit()
            subgraph = graph.subgraph(list(graph.filter_nodes(new_filt)))
            #subgraph = graph.subgraph(list(graph.filter_nodes(is_tl)))
            subgraph.copy_from_parent()"""
            
            #print("subgraph nodes :", subgraph.nodes())
            #print("subgraph ndata :", subgraph.ndata)
            #return self.layers['prediction_module'].predict(graph, device)
    
    
    
    
    

class Message_Module(nn.Module):
    def __init__(self, noisy, num_attention_heads, bias_before_aggregation, nonlinearity_before_aggregation, multidimensional_attention, data, use_attention, std_attention, resnet, use_message_module, state_vars, num_rels, num_bases, n_hidden, hidden_layers_size, in_feat , out_feat , bias = True, activation = F.relu, norm = True, dropout = False):
        super(Message_Module, self).__init__()    
        
        #print("in_feat", in_feat)
        self.noisy = noisy
        self.noisy = False
        self.num_attention_heads = num_attention_heads
        self.bias_before_aggregations = bias_before_aggregation
        self.nonlinearity_before_aggregation = nonlinearity_before_aggregation
        self.multidimensional_attention = multidimensional_attention
        self.data = data
        self.use_attention = use_attention
        self.std_attention = std_attention
        self.resnet = resnet
        self.use_message_module = use_message_module
        self.dropout = dropout
        self.state_vars = state_vars
        self.norm = norm
        self.hidden_layers_size = hidden_layers_size
        self.n_hidden = n_hidden
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation

        
        if self.use_message_module:
            self.message_weights = nn.ParameterList()  
            self.message_w_comps = nn.ParameterList()   
            self.message_biases = nn.ParameterList()   
            if self.noisy:
                self.sigma_message_weights = nn.ParameterList()  
                self.epsilon_message_weights = []
                if self.bias:
                    self.sigma_message_biases = nn.ParameterList() 
                    self.epsilon_message_biases = []  
                    
                    
            # sanity check
            if self.num_bases <= 0 or self.num_bases > self.num_rels:
                self.num_bases = self.num_rels        
 
                    
            
            # CREATE INFORMATION - MODULE
            
            
            
            if self.n_hidden == 0 :
                #UNIQUE LAYER 

                # weight bases in equation (3)

                self.weight_message_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                        self.out_feat))
                #print("weight size", self.weight_message_input.size())
                if self.num_bases < self.num_rels:
                    # linear combination coefficients in equation (3)
                    self.w_comp_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                #INITS
                nn.init.xavier_uniform_(self.weight_message_input,gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_rels:
                    nn.init.xavier_uniform_(self.w_comp_message_input,gain=nn.init.calculate_gain('relu'))            
                if self.bias:
                    self.bias_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))                
                    nn.init.uniform_(self.bias_message_input)   

                if self.noisy:
                    self.sigma_weight_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat,
                                                        self.out_feat).fill_(0.017))
                    self.register_buffer("epsilon_weight_message_input", torch.zeros(self.num_rels, self.in_feat, self.out_feat))
                    if self.bias:
                        self.sigma_bias_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat).fill_(0.017))
                    std = math.sqrt(3 / self.in_feat)
                    #nn.init.uniform(self.weight_message_input, -std, std)
                    if self.bias:
                        #nn.init.uniform(self.bias_message_input, -std, std)
                        self.register_buffer("epsilon_bias_message_input", torch.zeros(self.num_rels, self.out_feat))
                        #nn.init.uniform(self.sigma_bias_message_input, -std, std)
                    
                    
                
                
            else:

                #INPUT LAYER 

                # weight bases in equation (3)

                self.weight_message_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                        self.hidden_layers_size))


                #print("weight size", self.weight_message_input.size())
                if self.num_bases < self.num_rels:
                    # linear combination coefficients in equation (3)
                    self.w_comp_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                #INITS
                nn.init.xavier_uniform_(self.weight_message_input,gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_rels:
                    nn.init.xavier_uniform_(self.w_comp_message_input,gain=nn.init.calculate_gain('relu'))            
                if self.bias:
                    self.bias_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))                
                    nn.init.uniform_(self.bias_message_input)   


                if self.noisy:
                    self.sigma_weight_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat,
                                                        self.hidden_layers_size).fill_(0.017))
                    self.register_buffer("epsilon_weight_message_input", torch.zeros(self.num_rels, self.in_feat, self.hidden_layers_size))
                    if self.bias:
                        self.sigma_bias_message_input = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size).fill_(0.017))
                    std = math.sqrt(3 / self.in_feat)
                    #nn.init.uniform(self.weight_message_input, -std, std)
                    if self.bias:
                        #nn.init.uniform(self.bias_message_input, -std, std)
                        self.register_buffer("epsilon_bias_message_input", torch.zeros(self.num_rels, self.hidden_layers_size))
                        #nn.init.uniform(self.sigma_bias_message_input, -std, std)

                


                #HIDDEN LAYERS

                for _ in range(self.n_hidden -1):
                    # weight bases in equation (3)
                    weight_message = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.hidden_layers_size))
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    if self.bias:
                        bias_message = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))          
                        nn.init.uniform_(bias_message)              
                    # init trainable parameters
                    nn.init.xavier_uniform_(weight_message,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(w_comp,
                                                gain=nn.init.calculate_gain('relu'))            

                    self.message_weights.append(weight_message)
                    if self.num_bases < self.num_rels:            
                        self.message_w_comps.append(w_comp_message)
                    if self.bias:
                        self.message_biases.append(bias_message)
                        
                        
                    if self.noisy:
                        sigma_weight_message = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size,
                                                                                       self.hidden_layers_size).fill_(0.017))
                        self.register_buffer("epsilon_weight_message_" + str(n), torch.zeros(self.num_rels,
                                                                                            self.hidden_layers_size,
                                                                                            self.hidden_layers_size))
                        sigma_bias_message = nn.Parameter(torch.Tensor(self.num_rels, 
                                                                                     self.hidden_layers_size).fill_(0.017))
                        std = math.sqrt(3 / self.hidden_layers_size)
                        #nn.init.uniform(weight_message, -std, std)
                        #nn.init.uniform(bias_message, -std, std)
                        if self.bias:
                            self.register_buffer("epsilon_bias_message_" + str(n),
                                                 torch.zeros(self.num_rels,self.hidden_layers_size))
                            #nn.init.uniform(sigma_bias_message, -std, std)
                            
                        self.sigma_message_weights.append(sigma_weight_message)
                        self.epsilon_message_weights.append(getattr(self, "epsilon_weight_message_" + str(n)))
                        if self.bias:
                            self.sigma_message_biases.append(sigma_bias_message)
                            self.epsilon_message_biases.append(getattr(self, "epsilon_bias_message_" + str(n)))    
                            



                #OUTPUT LAYER 
                # weight bases in equation (3)

                self.weight_message_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                        self.out_feat))
                if self.num_bases < self.num_rels:
                    # linear combination coefficients in equation (3)
                    self.w_comp_message_output = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                #INITS
                nn.init.xavier_uniform_(self.weight_message_output,
                                        gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_rels:
                    nn.init.xavier_uniform_(self.w_comp_message_output,
                                            gain=nn.init.calculate_gain('relu'))            
                if self.bias:
                    self.bias_message_output = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))               
                    nn.init.uniform_(self.bias_message_output)     


                    
                if self.noisy:
                    self.sigma_weight_message_output = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size,
                                                                                   self.out_feat).fill_(0.017))
                    self.register_buffer("epsilon_weight_message_output", torch.zeros(self.rels, self.hidden_layers_size,
                                                                                        self.out_feat))
                    std = math.sqrt(3 / self.hidden_layers_size)
                    #nn.init.uniform(self.weight_message_output, -std, std)
                    #nn.init.uniform(self.bias_message_output, -std, std)
                    if self.bias:
                        self.sigma_bias_message_output = nn.Parameter(torch.Tensor(self.num_rels, 
                                                                                     self.out_feat).fill_(0.017))
                        self.register_buffer("epsilon_bias_message_output", torch.zeros(self.num_rels, self.out_feat))
                        #nn.init.uniform(self.sigma_bias_message_output, -std, std)


            if self.use_attention : 
                self.request_weights = nn.ParameterList()  
                self.request_w_comps = nn.ParameterList()   
                self.request_biases = nn.ParameterList()   
                self.attention_weights = nn.ParameterList()  
                self.attention_w_comps = nn.ParameterList()   
                self.attention_biases = nn.ParameterList()   
                if self.noisy:
                    self.sigma_request_weights = nn.ParameterList()  
                    self.sigma_attention_weights = nn.ParameterList() 
                    self.epsilon_request_weights = []
                    self.epsilon_attention_weights = []
                    if self.bias:
                        self.sigma_request_biases = nn.ParameterList() 
                        self.sigma_attention_biases = nn.ParameterList() 
                        self.epsilon_request_biases = []  
                        self.epsilon_attention_biases = []  
                # CREATE REQUEST - MODULE


                if self.n_hidden == 0 :


                    #UNIQUE LAYER 

                    # weight bases in equation (3)

                    self.weight_request_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                            self.out_feat))


                    #print("weight size", self.weight_request_input.size())
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_request_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_request_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.bias:
                        self.bias_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))                
                        nn.init.uniform_(self.bias_request_input)   

                    if self.noisy:
                        self.sigma_weight_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat,
                                                            self.out_feat).fill_(0.017))
                        self.register_buffer("epsilon_weight_request_input", torch.zeros(self.num_rels, self.in_feat, self.out_feat))
                        if self.bias:
                            self.sigma_bias_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat).fill_(0.017))
                        std = math.sqrt(3 / self.in_feat)
                        #nn.init.uniform(self.weight_request_input, -std, std)
                        if self.bias:
                            #nn.init.uniform(self.bias_request_input, -std, std)
                            self.register_buffer("epsilon_bias_request_input", torch.zeros(self.num_rels, self.out_feat))
                            #nn.init.uniform(self.sigma_bias_request_input, -std, std)




                else:


                    #INPUT LAYER 

                    # weight bases in equation (3)

                    self.weight_request_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                            self.hidden_layers_size))


                    #print("weight size", self.weight_request_input.size())
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_request_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_request_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.bias:
                        self.bias_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size))                
                        nn.init.uniform_(self.bias_request_input)   



                    if self.noisy:
                        self.sigma_weight_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat,
                                                            self.hidden_layers_size).fill_(0.017))
                        self.register_buffer("epsilon_weight_request_input", torch.zeros(self.num_rels, self.in_feat, self.hidden_layers_size))
                        if self.bias:
                            self.sigma_bias_request_input = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size).fill_(0.017))
                        std = math.sqrt(3 / self.in_feat)
                        #nn.init.uniform(self.weight_request_input, -std, std)
                        if self.bias:
                            #nn.init.uniform(self.bias_request_input, -std, std)
                            self.register_buffer("epsilon_bias_request_input", torch.zeros(self.num_rels, self.hidden_layers_size))
                            #nn.init.uniform(self.sigma_bias_request_input, -std, std)



                    #HIDDEN LAYERS

                    for _ in range(self.n_hidden -1):
                        # weight bases in equation (3)
                        weight_request = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                                self.hidden_layers_size))
                        if self.num_bases < self.num_rels:
                            # linear combination coefficients in equation (3)
                            w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                        if self.bias:
                            bias_request = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))          
                            nn.init.uniform_(bias_request)              
                        # init trainable parameters
                        nn.init.xavier_uniform_(weight_request,
                                                gain=nn.init.calculate_gain('relu'))
                        if self.num_bases < self.num_rels:
                            nn.init.xavier_uniform_(w_comp,
                                                    gain=nn.init.calculate_gain('relu'))            

                        self.request_weights.append(weight_request)
                        if self.num_bases < self.num_rels:            
                            self.request_w_comps.append(w_comp_request)
                        if self.bias:
                            self.request_biases.append(bias_request)

                        if self.noisy:
                            sigma_weight_request = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size,
                                                                                           self.hidden_layers_size).fill_(0.017))
                            self.register_buffer("epsilon_weight_request_" + str(n), torch.zeros(self.num_rels,
                                                                                                self.hidden_layers_size,
                                                                                                self.hidden_layers_size))
                            sigma_bias_request = nn.Parameter(torch.Tensor(self.num_rels, 
                                                                                         self.hidden_layers_size).fill_(0.017))
                            std = math.sqrt(3 / self.hidden_layers_size)
                            #nn.init.uniform(weight_request, -std, std)
                            #nn.init.uniform(bias_request, -std, std)
                            if self.bias:
                                self.register_buffer("epsilon_bias_request_" + str(n),
                                                     torch.zeros(self.num_rels,self.hidden_layers_size))
                                #nn.init.uniform(sigma_bias_request, -std, std)

                            self.sigma_request_weights.append(sigma_weight_request)
                            self.epsilon_request_weights.append(getattr(self, "epsilon_weight_request_" + str(n)))
                            if self.bias:
                                self.sigma_request_biases.append(sigma_bias_request)
                                self.epsilon_request_biases.append(getattr(self, "epsilon_bias_request_" + str(n)))    


                    #OUTPUT LAYER 
                    # weight bases in equation (3)

                    self.weight_request_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.out_feat))
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_request_output = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_request_output,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_request_output,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.bias:
                        self.bias_request_output = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))               
                        nn.init.uniform_(self.bias_request_output)     

                    
                    if self.noisy:
                        self.sigma_weight_request_output = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size,
                                                                                       self.out_feat).fill_(0.017))
                        self.register_buffer("epsilon_weight_request_output", torch.zeros(self.rels, self.hidden_layers_size,
                                                                                            self.out_feat))
                        std = math.sqrt(3 / self.hidden_layers_size)
                        #nn.init.uniform(self.weight_request_output, -std, std)
                        #nn.init.uniform(self.bias_request_output, -std, std)
                        if self.bias:
                            self.sigma_bias_request_output = nn.Parameter(torch.Tensor(self.num_rels, 
                                                                                         self.out_feat).fill_(0.017))
                            self.register_buffer("epsilon_bias_request_output", torch.zeros(self.num_rels, self.out_feat))
                            #nn.init.uniform(self.sigma_bias_request_output, -std, std)




                # CREATE ATTENTION - MODULE




                if self.n_hidden == 0 :


                    #UNIQUE LAYER 

                    # weight bases in equation (3)
                    self.weight_attention_input = nn.Parameter(torch.Tensor(self.num_bases, 2*self.hidden_layers_size,
                                                            self.out_feat if self.multidimensional_attention else self.num_attention_heads))
           


                    #print("weight size", self.weight_request_input.size())
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_attention_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_attention_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_attention_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.bias:
                        self.bias_attention_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat if self.multidimensional_attention else self.num_attention_heads))  
                        nn.init.uniform_(self.bias_attention_input)   

                    if self.noisy:
                        self.sigma_weight_attention_input = nn.Parameter(torch.Tensor(self.num_rels, 2*self.hidden_layers_size,
                                                            self.out_feat if self.multidimensional_attention else self.num_attention_heads).fill_(0.017))
                        self.register_buffer("epsilon_weight_attention_input", torch.zeros(self.num_rels, 2*self.hidden_layers_size,
                                                            self.out_feat if self.multidimensional_attention else self.num_attention_heads))
                        if self.bias:
                            self.sigma_bias_attention_input = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat if self.multidimensional_attention else self.num_attention_heads).fill_(0.017))
                        std = math.sqrt(3 / 2*self.hidden_layers_size)
                        #nn.init.uniform(self.weight_attention_input, -std, std)
                        if self.bias:
                            #nn.init.uniform(self.bias_attention_input, -std, std)
                            self.register_buffer("epsilon_bias_attention_input", torch.zeros(self.num_rels, self.out_feat if self.multidimensional_attention else self.num_attention_heads))
                            #nn.init.uniform(self.sigma_bias_attention_input, -std, std)





                else:

                    #INPUT LAYER 

                    # weight bases in equation (3)

                    self.weight_attention_input = nn.Parameter(torch.Tensor(self.num_bases, 2*self.hidden_layers_size,
                                                            self.hidden_layers_size))


                    #print("weight size", self.weight_attention_input.size())
                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_attention_input = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_attention_input,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_attention_input,
                                                gain=nn.init.calculate_gain('relu'))            
                    if self.bias:
                        self.bias_attention_input = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size))                
                        nn.init.uniform_(self.bias_attention_input)   



                    if self.noisy:
                        self.sigma_weight_attention_input = nn.Parameter(torch.Tensor(self.num_rels, 2*self.hidden_layers_size,
                                                            self.hidden_layers_size).fill_(0.017))
                        self.register_buffer("epsilon_weight_attention_input", torch.zeros(self.num_rels, 2*self.hidden_layers_size, self.hidden_layers_size))
                        if self.bias:
                            self.sigma_bias_attention_input = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size).fill_(0.017))
                        std = math.sqrt(3 / self.in_feat)
                        #nn.init.uniform(self.weight_attention_input, -std, std)
                        if self.bias:
                            #nn.init.uniform(self.bias_attention_input, -std, std)
                            self.register_buffer("epsilon_bias_attention_input", torch.zeros(self.num_rels, self.hidden_layers_size))
                            #nn.init.uniform(self.sigma_bias_attention_input, -std, std)




                    #HIDDEN LAYERS

                    for _ in range(self.n_hidden  -1):
                        # weight bases in equation (3)
                        weight_attention = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                                self.hidden_layers_size))
                        if self.num_bases < self.num_rels:
                            # linear combination coefficients in equation (3)
                            w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                        if self.bias:
                            bias_attention = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat))          
                            nn.init.uniform_(bias_attention)              
                        # init trainable parameters
                        nn.init.xavier_uniform_(weight_attention,
                                                gain=nn.init.calculate_gain('relu'))
                        if self.num_bases < self.num_rels:
                            nn.init.xavier_uniform_(w_comp,
                                                    gain=nn.init.calculate_gain('relu'))            

                        self.attention_weights.append(weight_attention)
                        if self.num_bases < self.num_rels:            
                            self.attention_w_comps.append(w_comp_attention)
                        if self.bias:
                            self.attention_biases.append(bias_attention)

                        if self.noisy:
                            sigma_weight_attention = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size,
                                                                                           self.hidden_layers_size).fill_(0.017))
                            self.register_buffer("epsilon_weight_attention_" + str(n), torch.zeros(self.num_rels,
                                                                                                self.hidden_layers_size,
                                                                                                self.hidden_layers_size))
                            sigma_bias_attention = nn.Parameter(torch.Tensor(self.num_rels, 
                                                                                         self.hidden_layers_size).fill_(0.017))
                            std = math.sqrt(3 / self.hidden_layers_size)
                            #nn.init.uniform(weight_attention, -std, std)
                            #nn.init.uniform(bias_attention, -std, std)
                            if self.bias:
                                self.register_buffer("epsilon_bias_attention_" + str(n),
                                                     torch.zeros(self.num_rels,self.hidden_layers_size))
                                #nn.init.uniform(sigma_bias_attention, -std, std)

                            self.sigma_attention_weights.append(sigma_weight_attention)
                            self.epsilon_attention_weights.append(getattr(self, "epsilon_weight_attention_" + str(n)))
                            if self.bias:
                                self.sigma_attention_biases.append(sigma_bias_attention)
                                self.epsilon_attention_biases.append(getattr(self, "epsilon_bias_attention_" + str(n)))    



                    #OUTPUT LAYER 
                    # weight bases in equation (3)
                    self.weight_attention_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.out_feat if self.multidimensional_attention else self.num_attention_heads))
         

                    if self.num_bases < self.num_rels:
                        # linear combination coefficients in equation (3)
                        self.w_comp_attention_output = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                    #INITS
                    nn.init.xavier_uniform_(self.weight_attention_output,
                                            gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_rels:
                        nn.init.xavier_uniform_(self.w_comp_attention_output,
                                                gain=nn.init.calculate_gain('relu'))   
                        
                        
                    if self.bias:
                        self.bias_attention_output = nn.Parameter(torch.Tensor(self.num_rels, self.out_feat if self.multidimensional_attention else self.num_attention_heads))                       
                        
                        nn.init.uniform_(self.bias_attention_output)   
                        

                    
                    if self.noisy:
                        self.sigma_weight_attention_output = nn.Parameter(torch.Tensor(self.num_rels, self.hidden_layers_size,
                                                                                       self.out_feat if self.multidimensional_attention else self.num_attention_heads).fill_(0.017))
                        self.register_buffer("epsilon_weight_attention_output", torch.zeros(self.rels, self.hidden_layers_size,
                                                                                            self.out_feat if self.multidimensional_attention else self.num_attention_heads))
                        std = math.sqrt(3 / self.hidden_layers_size)
                        #nn.init.uniform(self.weight_attention_output, -std, std)
                        #nn.init.uniform(self.bias_attention_output, -std, std)
                        if self.bias:
                            self.sigma_bias_attention_output = nn.Parameter(torch.Tensor(self.num_rels, 
                                                                                         self.out_feat if self.multidimensional_attention else self.num_attention_heads).fill_(0.017))
                            self.register_buffer("epsilon_bias_attention_output", torch.zeros(self.num_rels, self.out_feat if self.multidimensional_attention else self.num_attention_heads))
                            #nn.init.uniform(self.sigma_bias_attention_output, -std, std)




            
            
            
        
    def propagate(self, graph, device, testing = False):
           
        def message_func(edges):
            #INPUT LAYER 
            if self.use_message_module:
            
                # FORWARD MESSAGE 
                

            

                if self.n_hidden == 0 :
                

                    if self.num_bases < self.num_rels:
                        # generate all weights from bases (equation (3))
                        weight_message_input = self.weight_message_input.view(self.in_feat, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_message_input = torch.matmul(self.w_comp_message_input, weight_message_input).view(self.num_rels,
                                                                    self.in_feat, self.out_feat)#.to("cuda")     
                    else:
                        weight_message_input = self.weight_message_input#.to("cuda")    
                        #print("weight size", weight_message_input.size())

                    w_message_input = weight_message_input[edges.data['rel_type']]#.to("cuda")   
                    bias_message_input = self.bias_message_input[edges.data['rel_type']]

                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.epsilon_weight_message_input[edges.data['rel_type']].size(),device = device).normal_()
                        #e_w = torch.randn(self.epsilon_weight_prediction_input[nodes.data['node_type']].size())
                        w_message_input += self.sigma_weight_message_input[edges.data['rel_type']] * Variable(e_w)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.epsilon_bias_message_input[edges.data['rel_type']].size(),device = device).normal_()
                            #e_b = torch.randn(self.epsilon_bias_prediction_input[nodes.data['node_type']].size())
                            bias_message_input += self.sigma_bias_message_input[edges.data['rel_type']] * Variable(e_b)
                    #msg = torch.bmm(torch.cat((edges.src['state'].to(device), edges.src['hid'].to(device)),1).unsqueeze(1), w_message_input).squeeze()
                    msg = torch.bmm(torch.cat((tuple(edges.src[var].to(device) for var in self.data)),1).unsqueeze(1), w_message_input).squeeze()
                    

                        #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_message_input).squeeze()

                    if self.bias:
                        msg = msg + bias_message_input

                    if self.nonlinearity_before_aggregation:
                        msg = self.activation(msg) 
                        
                    if self.dropout :
                        msg = torch.nn.functional.dropout(msg, p=0.5, training=True, inplace=False) 
                    
                        
                    
                    
                    
                    
                else:

                    if self.num_bases < self.num_rels:
                        # generate all weights from bases (equation (3))
                        weight_message_input = self.weight_message_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        weight_message_input = torch.matmul(self.w_comp_message_input, weight_message_input).view(self.num_rels,
                                                                    self.in_feat, self.hidden_layers_size)#.to("cuda")     
                    else:
                        weight_message_input = self.weight_message_input#.to("cuda")    
                        #print("weight size", weight_message_input.size())

                    w_message_input = weight_message_input[edges.data['rel_type']]#.to("cuda")   
                    bias_message_input = self.bias_message_input[edges.data['rel_type']]

                    
                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.epsilon_weight_message_input[edges.data['rel_type']].size(),device = device).normal_()
                        #e_w = torch.randn(self.epsilon_weight_prediction_input[nodes.data['node_type']].size())
                        w_message_input += self.sigma_weight_message_input[edges.data['rel_type']] * Variable(e_w)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.epsilon_bias_message_input[nodes.data['rel_type']].size(),device = device).normal_()
                            #e_b = torch.randn(self.epsilon_bias_prediction_input[nodes.data['node_type']].size())
                            bias_message_input += self.sigma_bias_message_input[edges.data['rel_type']] * Variable(e_b)

                    #print("edges.src['h'].size()", edges.src['h'].size())
                    #msg = torch.bmm(torch.cat((edges.src['state'].to(device), edges.src['hid'].to(device)),1).unsqueeze(1), w_message_input).squeeze()
                    msg = torch.bmm(torch.cat((tuple(edges.src[var].to(device) for var in self.data)),1).unsqueeze(1), w_message_input).squeeze()

                    #print("msg", msg.size())
                        #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_message_input).squeeze()

                    if self.bias:
                        msg = msg + bias_message_input

                    msg = self.activation(msg)            
                    if self.dropout :
                        msg = torch.nn.functional.dropout(msg, p=0.5, training=True, inplace=False) 

                    #HIDDEN LAYERS            

                    for idx in range(self.n_hidden  -1):

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_message_hid = self.message_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_message_hid = torch.matmul(self.message_w_comps[idx], weight_message_hid).view(self.num_rels,
                                                                        self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                        else:
                            weight_message_hid = self.message_weights[idx]#.to("cuda")   

                        bias_message = self.message_biases[idx][edges.data['rel_type']]


                        w_message_hid = weight_message_hid[edges.data['rel_type']]#.to("cuda")   
                        
                        
                        if self.noisy and not testing:
                            e_w = torch.cuda.FloatTensor(self.epsilon_message_weights[idx][edges.data['rel_type']].size(),device = device).normal_()                            
                            #e_w = torch.randn(self.epsilon_prediction_weights[idx][nodes.data['node_type']].size())
                            w_message_hid += self.sigma_message_weights[idx][edges.data['rel_type']] * Variable(e_w).to(device)
                            if self.bias:
                                e_b = torch.cuda.FloatTensor(self.epsilon_message_biases[idx][edges.data['rel_type']].size(),device = device).normal_()    
                                #e_b = torch.randn(self.epsilon_prediction_biases[idx][nodes.data['node_type']].size())
                                bias_message_hid += self.sigma_message_biases[idx][edges.data['rel_type']] * Variable(e_b).to(device)
                                                        
                        
                        
                        
                        
                        msg = torch.bmm(msg.unsqueeze(1), w_message_hid).squeeze()

                        if self.bias:
                            msg = msg + bias_message

                        msg = self.activation(msg)      
                        if self.dropout :
                            msg = torch.nn.functional.dropout(msg, p=0.5, training=True, inplace=False)             
                    #OUTPUT LAYER 

                    if self.num_bases < self.num_rels:
                        # generate all weights from bases (equation (3))
                        weight_message_output = self.weight_message_output.view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        weight_message_output = torch.matmul(self.w_comp_message_output, weight_message_output).view(self.num_rels,
                                                                    self.hidden_layers_size, self.out_feat)#.to("cuda")     

                    else:
                        weight_message_output = self.weight_message_output#.to("cuda")    



                    w_message_output = weight_message_output[edges.data['rel_type']]#.to("cuda")   
                    bias_message_output = self.bias_message_output[edges.data['rel_type']]
                    
                    
                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.epsilon_weight_message_output[edges.data['rel_type']].size(),device = device).normal_()
                        #e_w = torch.randn(self.epsilon_weight_prediction_output[nodes.data['node_type']].size())
                        w_message_output += self.sigma_weight_message_output[edges.data['rel_type']] * Variable(e_w).to(device)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.epsilon_bias_message_output[edges.data['rel_type']].size(),device = device).normal_()
                            #e_b = torch.randn(self.epsilon_bias_prediction_output[nodes.data['node_type']].size())
                            bias_message_output += self.sigma_bias_message_output[edges.data['rel_type']] * Variable(e_b).to(device)
                    
                    
                    
                    
                    msg = torch.bmm(msg.unsqueeze(1), w_message_output).squeeze()



                    if self.bias:
                        msg = msg + bias_message_output            

                    if self.norm:
                        msg = msg * edges.data['norm']

                    if self.nonlinearity_before_aggregation:
                        msg = self.activation(msg)          
                        
                        
                    if self.dropout :
                        msg = torch.nn.functional.dropout(msg, p=0.5, training=True, inplace=False)  


                
            
                if self.use_attention : 
                    #TESTING START HERE

                    # FORWARD REQUEST


                    if self.n_hidden == 0 :

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_request_input = self.weight_request_input.view(self.in_feat, self.num_bases, self.out_feat)#.to("cuda")     
                            weight_request_input = torch.matmul(self.w_comp_request_input, weight_request_input).view(self.num_rels,
                                                                        self.in_feat, self.self.out_feat)#.to("cuda")     
                        else:
                            weight_request_input = self.weight_request_input#.to("cuda")    
                            #print("weight size", weight_request_input.size())


                        w_request_input = weight_request_input[edges.data['rel_type']]#.to("cuda")   
                        bias_request_input = self.bias_request_input[edges.data['rel_type']]

                        if self.noisy and not testing:
                            e_w = torch.cuda.FloatTensor(self.epsilon_weight_request_input[edges.data['rel_type']].size(),device = device).normal_()
                            #e_w = torch.randn(self.epsilon_weight_prediction_input[nodes.data['node_type']].size())
                            w_request_input += self.sigma_weight_request_input[edges.data['rel_type']] * Variable(e_w)
                            if self.bias:
                                e_b = torch.cuda.FloatTensor(self.epsilon_bias_request_input[edges.data['rel_type']].size(),device = device).normal_()
                                #e_b = torch.randn(self.epsilon_bias_prediction_input[nodes.data['node_type']].size())
                                bias_request_input += self.sigma_bias_request_input[edges.data['rel_type']] * Variable(e_b)
                        #msg = torch.bmm(torch.cat((edges.src['state'].to(device), edges.src['hid'].to(device)),1).unsqueeze(1), 

                        #rqst = torch.bmm(torch.cat((edges.dst['state'].to(device), edges.dst['hid'].to(device)),1).unsqueeze(1), w_request_input).squeeze()
                        rqst = torch.bmm(torch.cat((tuple(edges.src[var].to(device) for var in self.data)),1).unsqueeze(1), w_request_input).squeeze()

                            #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_request_input).squeeze()

                        if self.bias:
                            rqst = rqst + bias_request_input

                        rqst = self.activation(rqst)            
                        if self.dropout :
                            rqst = torch.nn.functional.dropout(rqst, p=0.5, training=True, inplace=False) 



                    else:

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_request_input = self.weight_request_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_request_input = torch.matmul(self.w_comp_request_input, weight_request_input).view(self.num_rels,
                                                                        self.in_feat, self.hidden_layers_size)#.to("cuda")     
                        else:
                            weight_request_input = self.weight_request_input#.to("cuda")    
                            #print("weight size", weight_request_input.size())

                        w_request_input = weight_request_input[edges.data['rel_type']]#.to("cuda")   
                        bias_request_input = self.bias_request_input[edges.data['rel_type']]

                        if self.noisy and not testing:
                            e_w = torch.cuda.FloatTensor(self.epsilon_weight_request_input[edges.data['rel_type']].size(),device = device).normal_()
                            #e_w = torch.randn(self.epsilon_weight_prediction_input[nodes.data['node_type']].size())
                            w_request_input += self.sigma_weight_request_input[edges.data['rel_type']] * Variable(e_w)
                            if self.bias:
                                e_b = torch.cuda.FloatTensor(self.epsilon_bias_request_input[edges.data['rel_type']].size(),device = device).normal_()
                                #e_b = torch.randn(self.epsilon_bias_prediction_input[nodes.data['node_type']].size())
                                bias_request_input += self.sigma_bias_request_input[edges.data['rel_type']] * Variable(e_b)
                        #rqst = torch.bmm(torch.cat((edges.dst['state'].to(device), edges.dst['hid'].to(device)),1).unsqueeze(1), w_request_input).squeeze()
                        rqst = torch.bmm(torch.cat((tuple(edges.src[var].to(device) for var in self.data)),1).unsqueeze(1), w_request_input).squeeze()

                            #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_request_input).squeeze()

                        if self.bias:
                            rqst = rqst + bias_request_input

                        rqst = self.activation(rqst)            
                        if self.dropout :
                            rqst = torch.nn.functional.dropout(rqst, p=0.5, training=True, inplace=False) 

                        #HIDDEN LAYERS            

                        for idx in range(self.n_hidden -1):

                            if self.num_bases < self.num_rels:
                                # generate all weights from bases (equation (3))
                                weight_request_hid = self.request_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                                weight_request_hid = torch.matmul(self.request_w_comps[idx], weight_request_hid).view(self.num_rels,
                                                                            self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                            else:
                                weight_request_hid = self.request_weights[idx]#.to("cuda")   

                            bias_request = self.request_biases[idx][edges.data['rel_type']]


                            w_request_hid = weight_request_hid[edges.data['rel_type']]#.to("cuda")
                            



                            if self.noisy and not testing:
                                e_w = torch.cuda.FloatTensor(self.epsilon_request_weights[idx][edges.data['rel_type']].size(),device = device).normal_()                            
                                #e_w = torch.randn(self.epsilon_prediction_weights[idx][nodes.data['node_type']].size())
                                w_request_hid += self.sigma_request_weights[idx][edges.data['rel_type']] * Variable(e_w).to(device)
                                if self.bias:
                                    e_b = torch.cuda.FloatTensor(self.epsilon_request_biases[idx][nodes.data['rel_type']].size(),device = device).normal_()    
                                    #e_b = torch.randn(self.epsilon_prediction_biases[idx][nodes.data['node_type']].size())
                                    bias_request_hid += self.sigma_request_biases[idx][edges.data['rel_type']] * Variable(e_b).to(device)


                            rqst = torch.bmm(rqst.unsqueeze(1), w_request_hid).squeeze()

                            if self.bias:
                                rqst = rqst + bias_request

                            rqst = self.activation(rqst)      
                            if self.dropout :
                                rqst = torch.nn.functional.dropout(rqst, p=0.5, training=True, inplace=False)             
                        #OUTPUT LAYER 

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_request_output = self.weight_request_output.view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_request_output = torch.matmul(self.w_comp_request_output, weight_request_output).view(self.num_rels,
                                                                        self.hidden_layers_size, self.out_feat)#.to("cuda")     

                        else:
                            weight_request_output = self.weight_request_output#.to("cuda")    



                        w_request_output = weight_request_output[edges.data['rel_type']]#.to("cuda")   
                        bias_request_output = self.bias_request_output[edges.data['rel_type']]

                        if self.noisy and not testing:
                            e_w = torch.cuda.FloatTensor(self.epsilon_weight_request_output[edges.data['rel_type']].size(),device = device).normal_()
                            #e_w = torch.randn(self.epsilon_weight_prediction_output[nodes.data['node_type']].size())
                            w_request_output += self.sigma_weight_request_output[edges.data['rel_type']] * Variable(e_w).to(device)
                            if self.bias:
                                e_b = torch.cuda.FloatTensor(self.epsilon_bias_request_output[edges.data['rel_type']].size(),device = device).normal_()
                                #e_b = torch.randn(self.epsilon_bias_prediction_output[nodes.data['node_type']].size())
                                bias_request_output += self.sigma_bias_request_output[edges.data['rel_type']] * Variable(e_b).to(device)

                        rqst = torch.bmm(rqst.unsqueeze(1), w_request_output).squeeze()



                        if self.bias:
                            rqst = rqst + bias_request_output            

                        if self.norm:
                            rqst = rqst * edges.data['norm']

                        rqst = self.activation(rqst)            
                        if self.dropout :
                            rqst = torch.nn.functional.dropout(rqst, p=0.5, training=True, inplace=False)  




                    # FORWARD ATTENTION
                    if self.n_hidden == 0 :

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_attention_input = self.weight_attention_input.view(2*self.hidden_layers_size, self.num_bases, self.out_feat if self.multidimensional_attention else self.num_attention_heads)#.to("cuda")     
                            weight_attention_input = torch.matmul(self.w_comp_attention_input, weight_attention_input).view(self.num_rels, 2*self.hidden_layers_size, self.out_feat if self.multidimensional_attention else self.num_attention_heads)#.to("cuda")     
                        else:
                            weight_attention_input = self.weight_attention_input#.to("cuda")    
                            #print("weight size", weight_attention_input.size())

                        w_attention_input = weight_attention_input[edges.data['rel_type']]#.to("cuda")   
                        bias_attention_input = self.bias_attention_input[edges.data['rel_type']]
                        #print("msg size", msg.size())
                        #print("rqst size", rqst.size())
                        #print("cat size", torch.cat((msg, rqst),1).size())
                        #print("self w_attention_input size", self.weight_attention_input.size())
                        #print("w_attention_input size", w_attention_input.size())
                        

                        if self.noisy and not testing:
                            e_w = torch.cuda.FloatTensor(self.epsilon_weight_attention_input[edges.data['rel_type']].size(),device = device).normal_()
                            #e_w = torch.randn(self.epsilon_weight_prediction_input[nodes.data['node_type']].size())
                            w_attention_input += self.sigma_weight_attention_input[edges.data['rel_type']] * Variable(e_w)
                            if self.bias:
                                e_b = torch.cuda.FloatTensor(self.epsilon_bias_attention_input[edges.data['rel_type']].size(),device = device).normal_()
                                #e_b = torch.randn(self.epsilon_bias_prediction_input[nodes.data['node_type']].size())
                                #print("e_b", e_b.size())
                                #print("sigma bias[reltype]", self.sigma_bias_attention_input[edges.data['rel_type']].size())
                                #print("sigma bias", self.sigma_bias_attention_input.size())
                                bias_attention_input += self.sigma_bias_attention_input[edges.data['rel_type']] * Variable(e_b)
                        #msg = torch.bmm(torch.cat((edges.src['state'].to(device), edges.src['hid'].to(device)),1).unsqueeze(1), 
                        att = torch.bmm(torch.cat((msg, rqst),1).unsqueeze(1), w_attention_input).squeeze()
                        #print("att", att.size())
                            #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_attention_input).squeeze()

                        if self.bias:
                            att = att + bias_attention_input

                        #print("att2", att.size())
                        att = self.activation(att)            
                        #if self.dropout :
                            #att = torch.nn.functional.dropout(att, p=0.5, training=True, inplace=False)   






                    else:
                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_attention_input = self.weight_attention_input.view(2*self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_attention_input = torch.matmul(self.w_comp_attention_input, weight_attention_input).view(self.num_rels,
                                                                        2*self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                        else:
                            weight_attention_input = self.weight_attention_input#.to("cuda")    
                            #print("weight size", weight_attention_input.size())

                        w_attention_input = weight_attention_input[edges.data['rel_type']]#.to("cuda")   
                        bias_attention_input = self.bias_attention_input[edges.data['rel_type']]
                        #print("msg size", msg.size())
                        #print("rqst size", rqst.size())
                        #print("cat size", torch.cat((msg, rqst),1).size())
                        #print("w_attention_input size", w_attention_input.size())
                        

                        if self.noisy and not testing:
                            e_w = torch.cuda.FloatTensor(self.epsilon_weight_attention_input[edges.data['rel_type']].size(),device = device).normal_()
                            #e_w = torch.randn(self.epsilon_weight_prediction_input[nodes.data['node_type']].size())
                            w_attention_input += self.sigma_weight_attention_input[edges.data['rel_type']] * Variable(e_w)
                            if self.bias:
                                e_b = torch.cuda.FloatTensor(self.epsilon_bias_attention_input[edges.data['rel_type']].size(),device = device).normal_()
                                #e_b = torch.randn(self.epsilon_bias_prediction_input[nodes.data['node_type']].size())
                                bias_attention_input += self.sigma_bias_attention_input[edges.data['rel_type']] * Variable(e_b)
                        #msg = torch.bmm(torch.cat((edges.src['state'].to(device), edges.src['hid'].to(device)),1).unsqueeze(1), 
                        att = torch.bmm(torch.cat((msg, rqst),1).unsqueeze(1), w_attention_input).squeeze()

                            #(edges.src['current_phases'].to("cuda"),edges.src['cycle_durations'].to("cuda"), edges.src['short_current_phases'].to("cuda"), edges.src['short_cycle_durations'].to("cuda"),edges.src['h'].to("cuda")),1).unsqueeze(1), w_attention_input).squeeze()
                        if self.bias:
                            att = att + bias_attention_input

                        att = self.activation(att)            
                        if self.dropout :
                            att = torch.nn.functional.dropout(att, p=0.5, training=True, inplace=False) 

                        #HIDDEN LAYERS            

                        for idx in range(self.n_hidden -1):

                            if self.num_bases < self.num_rels:
                                # generate all weights from bases (equation (3))
                                weight_attention_hid = self.attention_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                                weight_attention_hid = torch.matmul(self.attention_w_comps[idx], weight_attention_hid).view(self.num_rels,
                                                                            self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                            else:
                                weight_attention_hid = self.attention_weights[idx]#.to("cuda")   

                            bias_attention = self.attention_biases[idx][edges.data['rel_type']]


                            w_attention_hid = weight_attention_hid[edges.data['rel_type']]#.to("cuda") 


                            if self.noisy and not testing:
                                e_w = torch.cuda.FloatTensor(self.epsilon_attention_weights[idx][edges.data['rel_type']].size(),device = device).normal_()                            
                                #e_w = torch.randn(self.epsilon_prediction_weights[idx][nodes.data['node_type']].size())
                                w_attention_hid += self.sigma_attention_weights[idx][edges.data['rel_type']] * Variable(e_w).to(device)
                                if self.bias:
                                    e_b = torch.cuda.FloatTensor(self.epsilon_attention_biases[idx][edges.data['rel_type']].size(),device = device).normal_()    
                                    #e_b = torch.randn(self.epsilon_prediction_biases[idx][nodes.data['node_type']].size())
                                    bias_attention_hid += self.sigma_attention_biases[idx][edges.data['rel_type']] * Variable(e_b).to(device)


                            att = torch.bmm(att.unsqueeze(1), w_attention_hid).squeeze()

                            if self.bias:
                                att = att + bias_attention

                            att = self.activation(att)      
                            if self.dropout :
                                att = torch.nn.functional.dropout(att, p=0.5, training=True, inplace=False)             
                        #OUTPUT LAYER 

                        if self.num_bases < self.num_rels:
                            # generate all weights from bases (equation (3))
                            weight_attention_output = self.weight_attention_output.view(self.hidden_layers_size, self.num_bases, self.out_feat if self.multidimensional_attention else self.num_attention_heads)#.to("cuda")     
                            weight_attention_output = torch.matmul(self.w_comp_attention_output, weight_attention_output).view(self.num_rels, self.hidden_layers_size, self.out_feat if self.multidimensional_attention else self.num_attention_heads)#.to("cuda")     

                        else:
                            weight_attention_output = self.weight_attention_output#.to("cuda")    



                        w_attention_output = weight_attention_output[edges.data['rel_type']]#.to("cuda")   
                        bias_attention_output = self.bias_attention_output[edges.data['rel_type']]

                        if self.noisy and not testing:
                            e_w = torch.cuda.FloatTensor(self.epsilon_weight_attention_output[edges.data['rel_type']].size(),device = device).normal_()
                            #e_w = torch.randn(self.epsilon_weight_prediction_output[nodes.data['node_type']].size())
                            w_attention_output += self.sigma_weight_attention_output[edges.data['rel_type']] * Variable(e_w).to(device)
                            if self.bias:
                                e_b = torch.cuda.FloatTensor(self.epsilon_bias_attention_output[edges.data['rel_type']].size(),device = device).normal_()
                                #e_b = torch.randn(self.epsilon_bias_prediction_output[nodes.data['node_type']].size())
                                bias_attention_output += self.sigma_bias_attention_output[edges.data['rel_type']] * Variable(e_b).to(device)

                        att = torch.bmm(att.unsqueeze(1), w_attention_output).squeeze()



                        if self.bias:
                            att = att + bias_attention_output            

                        if self.norm:
                            att = att * edges.data['norm']


                        att = self.activation(att)            
                        #if self.dropout :
                        #    att = torch.nn.functional.dropout(att, p=0.5, training=True, inplace=False)  



                    return {'msg': msg, 'att' : att}            


                else: 
                    return {'msg' : msg}
                #TESTING START HERE
                
    
        
        
            else:
                msg = edges.src['short_cycle_durations']
                return {'msg': -msg}

        """
        if not self.use_message_module:
            def reduce_func(nodes):
                hid , _ = nodes.mailbox['msg'].min(0)
                print("hid size", hid.size())
                return {'hid', hid}
        """
        def reduce_func(nodes):
            # AGGREGATION OF ATTENTION 
            # REDUCE FUNCTION BATCHES NODES OF SAME IN-DEGREES TOGETHER 
            
            
            if self.use_attention : 
                att_w = nodes.mailbox['att']
                #print("att_w before", att_w.size(), att_w)
                if self.std_attention:
                    att_w = F.softmax(att_w,dim = 1)
                mailbox = nodes.mailbox['msg']
                    
                #print("att_w", att_w.size(), att_w, "nodes.mailbox[msg]",mailbox.size(), mailbox)
                if not self.multidimensional_attention:
                    mailbox = mailbox.unsqueeze(2).repeat(1,1,self.num_attention_heads,1)
                    #print('new message size', mailbox.size(), mailbox)
                    att_w = att_w.unsqueeze(3).expand_as(mailbox)
                    #print("new att size", att_w.size(), att_w)
                agg_msg = torch.sum( att_w * mailbox, dim = 1)
                #print("agg_msg", agg_msg.size(), agg_msg)
                
            
            else:
                agg_msg = torch.sum(nodes.mailbox['msg'], dim = 1)

            
            if not self.nonlinearity_before_aggregation:
                agg_msg = self.activation(agg_msg)
                
            return {'agg_msg' : agg_msg}
            #return{'agg_msg' : torch.mean(nodes.mailbox['msg'], dim = 1)}

        def apply_func(nodes):
            #print("1st agg msg size :", nodes.data["agg_msg"].size())
            pass
        
        
        if self.use_message_module: 
            graph.update_all(message_func = message_func, reduce_func = reduce_func, apply_node_func = apply_func)     

            
        else:
            graph.update_all(message_func = message_func, reduce_func = fn.max(msg='msg',out='hid'), apply_node_func = apply_func)  
        #print("went through message layer")
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class Aggregation_Module(nn.Module):

    def __init__(self, noisy, use_attention, num_attention_heads, multidimensional_attention, resnet, use_aggregation_module, num_nodes_types, num_bases, n_hidden, hidden_layers_size, in_feat , out_feat , bias = True, activation = F.relu, norm = False, dropout = False):

        super(Aggregation_Module, self).__init__()   
        self.noisy = noisy
        self.use_attention = use_attention
        self.num_attention_heads = num_attention_heads
        self.multidimensional_attention = multidimensional_attention
        self.use_aggregation_module = use_aggregation_module
        self.resnet = resnet
        self.dropout = dropout
        self.norm = norm
        self.hidden_layers_size = hidden_layers_size
        self.n_hidden = n_hidden
        
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_nodes_types = num_nodes_types
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation

        
        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_nodes_types:
            self.num_bases = self.num_nodes_types   
            
        # RNN - GRU 
        
        """
        r=σ(Wir​x+bir+Whr​h+bhr)
        z=σ(Wiz​x+biz+Whz​h+bhz)
        n=tanh(Win​x+bin+r∗(Whn​h+bhn))
        h′=(1−z)∗n+z∗h
        """
        
        # MAPPING FROM CONCATENATED MULTI-HEAD ATTENTION MECHANISMS RESULTS TO THE ORIGINAL EMBEDDING SIZE 
        if self.use_attention and not self.multidimensional_attention:
                self.weight_att_head_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_attention_heads*self.in_feat, self.in_feat))  
                nn.init.xavier_uniform_(self.weight_att_head_aggregation, gain=nn.init.calculate_gain('relu')) 
                if self.num_bases < self.num_nodes_types:
                    self.w_comp_att_head_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))   
                    nn.init.xavier_uniform_(self.w_comp_att_head_aggregation, gain=nn.init.calculate_gain('relu'))       
                if self.bias:
                    self.bias_att_head_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, self.out_feat))
                    nn.init.uniform_(self.bias_att_head_aggregation)   
            # CREATE AND INIT WEIGHTS
            
        if self.use_aggregation_module : 
            self.aggregation_weights = nn.ParameterList()  
            self.aggregation_w_comps = nn.ParameterList()   
            self.aggregation_biases = nn.ParameterList()   
            self.weights_input_aggregation = nn.ParameterList()  
            self.weights_hidden_aggregation = nn.ParameterList()  
            if self.num_bases < self.num_nodes_types:       
                self.w_comps_input_aggregation = nn.ParameterList()  
                self.w_comps_hidden_aggregation = nn.ParameterList()  
            if self.bias:
                self.biases_input_aggregation = nn.ParameterList()  
                self.biases_hidden_aggregation = nn.ParameterList()  
                
                
                
                
            if self.n_hidden == 0 : 
                # UNIQUE LAYER 


                # BETTER PARRAL
                self.weight_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, 3*self.out_feat))
                self.weight_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_bases, self.out_feat, 3*self.out_feat))            
                nn.init.xavier_uniform_(self.weight_input_aggregation_input, gain=nn.init.calculate_gain('relu'))            
                nn.init.xavier_uniform_(self.weight_hidden_aggregation_input, gain=nn.init.calculate_gain('relu'))            


                if self.num_bases < self.num_nodes_types:            
                    self.w_comp_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    self.w_comp_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.self.w_comp_input_aggregation_input, gain=nn.init.calculate_gain('relu'))            
                    nn.init.xavier_uniform_(self.self.w_comp_hidden_aggregation_input, gain=nn.init.calculate_gain('relu'))                

                if self.bias:
                    self.bias_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.out_feat))
                    self.bias_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.out_feat))
                    nn.init.uniform_(self.bias_input_aggregation_input)   
                    nn.init.uniform_(self.bias_hidden_aggregation_input)   



            else:


                # INPUT LAYER 


                # BETTER PARRAL
                self.weight_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, 3*self.hidden_layers_size))
                self.weight_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size, 3*self.hidden_layers_size))            
                nn.init.xavier_uniform_(self.weight_input_aggregation_input, gain=nn.init.calculate_gain('relu'))            
                nn.init.xavier_uniform_(self.weight_hidden_aggregation_input, gain=nn.init.calculate_gain('relu'))            


                if self.num_bases < self.num_nodes_types:            
                    self.w_comp_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    self.w_comp_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.self.w_comp_input_aggregation_input, gain=nn.init.calculate_gain('relu'))            
                    nn.init.xavier_uniform_(self.self.w_comp_hidden_aggregation_input, gain=nn.init.calculate_gain('relu'))                

                if self.bias:
                    self.bias_input_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.hidden_layers_size))
                    self.bias_hidden_aggregation_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.hidden_layers_size))
                    nn.init.uniform_(self.bias_input_aggregation_input)   
                    nn.init.uniform_(self.bias_hidden_aggregation_input)   


                #HIDDEN LAYERS


                for _ in range(self.n_hidden -1):

                    # BETTER PARRAL
                    weight_input_aggregation = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size, 3*self.hidden_layers_size))
                    weight_hidden_aggregation = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size, 3*self.hidden_layers_size))            
                    nn.init.xavier_uniform_(weight_input_aggregation, gain=nn.init.calculate_gain('relu'))            
                    nn.init.xavier_uniform_(weight_hidden_aggregation, gain=nn.init.calculate_gain('relu'))    
                    self.weights_input_aggregation.append(weight_input_aggregation)
                    self.weights_hidden_aggregation.append(weight_hidden_aggregation)                


                    if self.num_bases < self.num_nodes_types:            
                        w_comp_input_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                        w_comp_hidden_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                        nn.init.xavier_uniform_(w_comp_input_aggregation, gain=nn.init.calculate_gain('relu'))            
                        nn.init.xavier_uniform_(w_comp_hidden_aggregation, gain=nn.init.calculate_gain('relu')) 
                        self.w_comps_input_aggregation.append(w_comp_input_aggregation)
                        self.w_comps_hidden_aggregation.append(w_com_hidden_aggregation)

                    if self.bias:
                        bias_input_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.hidden_layers_size))
                        bias_hidden_aggregation = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.hidden_layers_size))
                        nn.init.uniform_(bias_input_aggregation)   
                        nn.init.uniform_(bias_hidden_aggregation)   
                        self.biases_input_aggregation.append(bias_input_aggregation)
                        self.biases_hidden_aggregation.append(bias_hidden_aggregation)


                # OUTPUT LAYER 


                # BETTER PARRAL
                self.weight_input_aggregation_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size, 3*self.out_feat))
                self.weight_hidden_aggregation_output = nn.Parameter(torch.Tensor(self.num_bases, self.out_feat, 3*self.out_feat))            
                nn.init.xavier_uniform_(self.weight_input_aggregation_output, gain=nn.init.calculate_gain('relu'))            
                nn.init.xavier_uniform_(self.weight_hidden_aggregation_output, gain=nn.init.calculate_gain('relu'))            


                if self.num_bases < self.num_nodes_types:            
                    self.w_comp_input_aggregation_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    self.w_comp_hidden_aggregation_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.self.w_comp_input_aggregation_output, gain=nn.init.calculate_gain('relu'))            
                    nn.init.xavier_uniform_(self.self.w_comp_hidden_aggregation_output, gain=nn.init.calculate_gain('relu'))                

                if self.bias:
                    self.bias_input_aggregation_output = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.out_feat))
                    self.bias_hidden_aggregation_output = nn.Parameter(torch.Tensor(self.num_nodes_types, 3*self.out_feat))
                    nn.init.uniform_(self.bias_input_aggregation_output)   
                    nn.init.uniform_(self.bias_hidden_aggregation_output)   


                
                
                
                
        #print('params', [param.size() for param in self.parameters()])
                
                
                
                

        
    def aggregate(self, graph, device, testing = False):
        #print("yololo2")
        def message_func(edges):
            pass
        def reduce_func(nodes):
            pass
            
        def apply_func(nodes):
            
            
            agg_msg = nodes.data['agg_msg']
            
            #print("agg_msg size :", agg_msg.size())

            # MAPPING FROM CONCATENATED RESULTS OF MULTIPLE ATTENTION HEAD MECHANISMS TO ORIGINAL EMBEDDING DIM
            if not self.multidimensional_attention and self.use_attention:
                if self.num_bases < self.num_nodes_types:
                    weight_att_head_aggregation = self.weight_att_head_aggregation.view(self.num_attention_heads * self.in_feat, self.num_bases, self.in_feat) 
                    weight_att_head_aggregation = torch.matmul(self.w_comp_att_head_aggregation, weight_att_head_aggregation).view(self.num_nodes_types, self.num_attention_heads * self.in_feat, self.in_feat)  
                else:
                    weight_att_head_aggregation = self.weight_att_head_aggregation 

                    
                #print("weight_att_head_aggregation", weight_att_head_aggregation.size())
                weight_att_head_aggregation = weight_att_head_aggregation[nodes.data['node_type']]
                bias_att_head_aggregation = self.bias_att_head_aggregation[nodes.data['node_type']]
                agg_msg = agg_msg.view(-1,self.num_attention_heads*self.in_feat)
                agg_msg = torch.bmm(agg_msg.unsqueeze(1), weight_att_head_aggregation).squeeze()
                if self.bias:
                    agg_msg = agg_msg + bias_att_head_aggregation
                agg_msg = self.activation(agg_msg)            
                if self.dropout :
                    agg_msg = torch.nn.functional.dropout(agg_msg, p=0.5, training=True, inplace=False) 



            
            
            
            
            # OPTIONNAL GRU LAYER (OPTION TO MAKE RESIDUAL CONNECTIONS <- self.resnet)

            if self.use_aggregation_module: 
                if self.n_hidden == 0 : 


                    
                    
                    # FORWARD UNIQUE
                    if self.num_bases < self.num_nodes_types:

                        weight_input_aggregation_input = self.weight_input_aggregation_input.view(self.in_feat, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_input_aggregation_input = torch.matmul(self.w_comp_input_aggregation_input, weight_input_aggregation_input).view(self.num_nodes_types,
                                                                    self.in_feat, 3*self.out_feat)#.to("cuda")     

                        weight_hidden_aggregation_input = self.weight_hidden_aggregation_input.view(self.out_feat, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_hidden_aggregation_input = torch.matmul(self.w_comp_hidden_aggregation_input, weight_hidden_aggregation_input).view(self.num_nodes_types,
                                                                    self.out_feat, 3*self.out_feat)#.to("cuda")

                    else:
                        weight_input_aggregation_input = self.weight_input_aggregation_input     
                        weight_hidden_aggregation_input = self.weight_hidden_aggregation_input



                    w_input_aggregation_input = weight_input_aggregation_input[nodes.data['node_type']]#.to("cuda")   
                    w_hidden_aggregation_input = weight_hidden_aggregation_input[nodes.data['node_type']]

                    if self.bias:
                        bias_input_aggregation_input = self.bias_input_aggregation_input[nodes.data['node_type']]
                        bias_hidden_aggregation_input = self.bias_hidden_aggregation_input[nodes.data['node_type']]

    
                    #print("agg msg size :", nodes.data["agg_msg"].size())
                    #print("w size :", weight_input_aggregation_input[nodes.data['node_type']].size())
                    
                    gate_x = torch.bmm(agg_msg.to(device).unsqueeze(1), w_input_aggregation_input).squeeze()
                    gate_h = torch.bmm(nodes.data['hid'].to(device).unsqueeze(1), w_hidden_aggregation_input).squeeze()

                    if self.bias:
                        #print("gate x size :", gate_x.size())
                        #print("bias size :", bias_input_aggregation_input.size())
                        gate_x += bias_input_aggregation_input
                        gate_h += bias_hidden_aggregation_input

                    #print("new gate x size :", gate_x.size())

                    i_r, i_i, i_n = gate_x.chunk(3, 1)
                    h_r, h_i, h_n = gate_h.chunk(3, 1)                

                    resetgate = torch.sigmoid(i_r + h_r)
                    inputgate = torch.sigmoid(i_i + h_i)
                    newgate = torch.tanh(i_n + (resetgate * h_n))



                    if self.resnet:
                        hy = nodes.data['hid'].to(device) + (newgate + inputgate * (nodes.data['hid'].to(device) - newgate))    
                    else:
                        hy = newgate + inputgate * (nodes.data['hid'].to(device) - newgate)
                        
                else:


                    # FORWARD INPUT
                    if self.num_bases < self.num_nodes_types:

                        weight_input_aggregation_input = self.weight_input_aggregation_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        weight_input_aggregation_input = torch.matmul(self.w_comp_input_aggregation_input, weight_input_aggregation_input).view(self.num_nodes_types,
                                                                    self.in_feat, 3*self.hidden_layers_size)#.to("cuda")     

                        weight_hidden_aggregation_input = self.weight_hidden_aggregation_input.view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        weight_hidden_aggregation_input = torch.matmul(self.w_comp_hidden_aggregation_input, weight_hidden_aggregation_input).view(self.num_nodes_types,
                                                                    self.hidden_layers_size, 3*self.hidden_layers_size)#.to("cuda")

                    else:
                        weight_input_aggregation_input = self.weight_input_aggregation_input     
                        weight_hidden_aggregation_input = self.weight_hidden_aggregation_input



                    w_input_aggregation_input = weight_input_aggregation_input[nodes.data['node_type']]#.to("cuda")   
                    w_hidden_aggregation_input = weight_hidden_aggregation_input[nodes.data['node_type']]
                    if self.bias:
                        bias_input_aggregation_input = self.bias_input_aggregation_input[nodes.data['node_type']]
                        bias_hidden_aggregation_input = self.bias_hidden_aggregation_input[nodes.data['node_type']]


                    gate_x = torch.bmm(agg_msg.to(device).unsqueeze(1), w_input_aggregation_input).squeeze()
                    gate_h = torch.bmm(nodes.data['memory_input'].to(device).unsqueeze(1), w_hidden_aggregation_input).squeeze()
                    if self.bias:
                        gate_h += bias_hidden_aggregation_input
                        gate_x += bias_input_aggregation_input

                    i_r, i_i, i_n = gate_x.chunk(3, 1)
                    h_r, h_i, h_n = gate_h.chunk(3, 1)                

                    resetgate = torch.sigmoid(i_r + h_r)
                    inputgate = torch.sigmoid(i_i + h_i)
                    newgate = torch.tanh(i_n + (resetgate * h_n))

                    if self.resnet:
                        hy = nodes.data['memory_input'].to(device) + (newgate + inputgate * (nodes.data['memory_input'].to(device)- newgate))    
                    else:
                        hy = newgate + inputgate * (nodes.data['memory_input'].to(device) - newgate)


                    graph.ndata.update({'memory_input': hy})                



                    # FORWARD HIDDEN

                    for idx in range(self.n_hidden-1):
                        if self.num_bases < self.num_nodes_types:


                            weight_input_aggregation = self.weights_input_aggregation[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_input_aggregation = torch.matmul(self.w_comps_input_aggregation[idx], weight_input_aggregation).view(self.num_nodes_types,
                                                                        self.hidden_layers_size, 3*self.hidden_layers_size)#.to("cuda")     

                            weight_hidden_aggregation = self.weights_hidden_aggregation[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_hidden_aggregation = torch.matmul(self.w_comps_hidden_aggregation[idx], weight_hidden_aggregation).view(self.num_nodes_types,
                                                                        self.hidden_layers_size, 3*self.hidden_layers_size)#.to("cuda")

                        else:
                            weight_input_aggregation = self.weight_input_aggregation_input     
                            weight_hidden_aggregation = self.weight_hidden_aggregation_input



                        w_input_aggregation = weight_input_aggregation[nodes.data['node_type']]#.to("cuda")   
                        w_hidden_aggregation = weight_hidden_aggregation[nodes.data['node_type']]
                        if self.bias:
                            bias_input_aggregation = self.bias_input_aggregation[nodes.data['node_type']]
                            bias_hidden_aggregation = self.bias_hidden_aggregation[nodes.data['node_type']]


                        gate_x = torch.bmm(hy.unsqueeze(1), w_input_aggregation).squeeze()
                        gate_h = torch.bmm(nodes.data[str('memory_' + str(idx))].unsqueeze(1), w_hidden_aggregation).squeeze()
                        if self.bias:
                            gate_x += bias_input_aggregation
                            gate_h += bias_hidden_aggregation

                        i_r, i_i, i_n = gate_x.chunk(3, 1)
                        h_r, h_i, h_n = gate_h.chunk(3, 1)                

                        resetgate = torch.sigmoid(i_r + h_r)
                        inputgate = torch.sigmoid(i_i + h_i)
                        newgate = torch.tanh(i_n + (resetgate * h_n))


                        if self.resnet:
                            hy = nodes.data[str('memory_' + str(idx))] + (newgate + inputgate * (nodes.data[str('memory_' + str(idx))] - newgate))   
                        else:
                            hy = newgate + inputgate * (nodes.data[str('memory_' + str(idx))] - newgate)

                        graph.ndata.update({str('memory_' + str(idx)): hy})



                    # FORWARD OUTPUT
                    if self.num_bases < self.num_nodes_types:

                        weight_input_aggregation_output = self.weight_input_aggregation_output.view(self.hidden_layers_size, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_input_aggregation_output = torch.matmul(self.w_comp_input_aggregation_output, weight_input_aggregation_output).view(self.num_nodes_types,
                                                                    self.hidden_layers_size, 3*self.out_feat)#.to("cuda")     

                        weight_hidden_aggregation_output = self.weight_hidden_aggregation_output.view(self.out_feat, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_hidden_aggregation_output = torch.matmul(self.w_comp_hidden_aggregation_output, weight_hidden_aggregation_output).view(self.num_nodes_types,
                                                                    self.out_feat, 3*self.out_feat)#.to("cuda")

                    else:
                        weight_input_aggregation_output = self.weight_input_aggregation_output     
                        weight_hidden_aggregation_output = self.weight_hidden_aggregation_output



                    w_input_aggregation_output = weight_input_aggregation_output[nodes.data['node_type']]#.to("cuda")   
                    w_hidden_aggregation_output = weight_hidden_aggregation_output[nodes.data['node_type']]
                    if self.bias:
                        bias_input_aggregation_output = self.bias_input_aggregation_output[nodes.data['node_type']]
                        bias_hidden_aggregation_output = self.bias_hidden_aggregation_output[nodes.data['node_type']]


                    gate_x = torch.bmm(hy.unsqueeze(1), w_input_aggregation_output).squeeze()
                    gate_h = torch.bmm(nodes.data['hid'].unsqueeze(1), w_hidden_aggregation_output).squeeze()
                    if self.bias:
                        gate_x += bias_input_aggregation_output
                        gate_h += bias_hidden_aggregation_output

                    i_r, i_i, i_n = gate_x.chunk(3, 1)
                    h_r, h_i, h_n = gate_h.chunk(3, 1)                

                    resetgate = torch.sigmoid(i_r + h_r)
                    inputgate = torch.sigmoid(i_i + h_i)
                    newgate = torch.tanh(i_n + (resetgate * h_n))
             

                    if self.resnet:
                        hy = nodes.data[str('hid')] + (newgate + inputgate * (nodes.data[str('hid')] - newgate))   
                    else:
                        hy = newgate + inputgate * (nodes.data[str('hid')] - newgate)
              


            else:
                if self.resnet:
                    hy = nodes.data['hid'].to(device) + agg_msg.to(device)
                else:
                    hy = agg_msg.to(device)
            return {'hid' : hy}      


        
        graph.update_all(message_func = message_func, reduce_func = reduce_func, apply_node_func = apply_func)             



            
            
            
class Prediction_Module(nn.Module):

    def __init__(self, dueling, policy, noisy, share_initial_params_between_actions, gaussian_mixture, n_gaussians, value_model_based, n_actions, rl_learner_type, use_message_module, num_nodes_types, num_bases, n_hidden, hidden_layers_size, in_feat , out_feat , bias = True, activation = F.relu, norm = True, dropout = False):
        super(Prediction_Module, self).__init__() 
        self.dueling = dueling
        self.policy = policy
        self.bias = bias
        self.noisy = noisy
        self.share_initial_params_between_actions = share_initial_params_between_actions
        self.gaussian_mixture = gaussian_mixture
        self.n_gaussians = n_gaussians
        self.value_model_based = value_model_based
        self.n_actions = n_actions
        self.rl_learner_type = rl_learner_type
        #self.rl_learner_type = 'critic'
        if self.policy != 'binary' :
            self.rl_learner_type = 'critic'
            
        self.use_message_module = use_message_module
        self.dropout = dropout
        self.norm = norm
        self.hidden_layers_size = hidden_layers_size
        self.n_hidden = n_hidden   
        self.in_feat = in_feat
        self.out_feat = out_feat
        #if self.dueling:
            #self.out_feat += 1
        if self.rl_learner_type == "Q_Learning":
            self.out_feat = self.n_actions
            self.prediction_weights = nn.ParameterList()  
            self.prediction_w_comps = nn.ParameterList()   
            self.prediction_biases = nn.ParameterList()   
            if self.noisy:
                self.sigma_prediction_weights = nn.ParameterList()  
                self.epsilon_prediction_weights = []
                if self.bias:
                    self.sigma_prediction_biases = nn.ParameterList() 
                    self.epsilon_prediction_biases = []

        if 'critic' in self.rl_learner_type.lower():
            self.critic_out_feat = 1
            self.critic_prediction_weights = nn.ParameterList()  
            self.critic_prediction_w_comps = nn.ParameterList()   
            self.critic_prediction_biases = nn.ParameterList()  
            if self.noisy:
                self.critic_sigma_prediction_weights = nn.ParameterList()  
                self.critic_epsilon_prediction_weights = []
                if self.bias:
                    self.critic_sigma_prediction_biases = nn.ParameterList() 
                    self.critic_epsilon_prediction_biases = []

        if 'actor' in self.rl_learner_type.lower():
            self.actor_out_feat = self.n_actions
            self.actor_prediction_weights = nn.ParameterList()  
            self.actor_prediction_w_comps = nn.ParameterList()   
            self.actor_prediction_biases = nn.ParameterList() 
            if self.noisy:
                self.actor_sigma_prediction_weights = nn.ParameterList()  
                self.actor_epsilon_prediction_weights = []
                if self.bias:
                    self.actor_sigma_prediction_biases = nn.ParameterList() 
                    self.actor_epsilon_prediction_biases = []
        
        self.num_nodes_types = num_nodes_types
        self.num_bases = num_bases
        self.activation = activation

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_nodes_types:
            self.num_bases = self.num_nodes_types



        if self.n_hidden == 0:
            
            """
            if self.rl_learner_type == "Q_Learning":    
                #self.q_noisy_linear = NoisyFactorizedLinear(self.in_feat, self.out_feat, sigma_zero=0.4, bias=True)
                self.q_noisy_linear = NoisyLinear(self.in_feat, self.out_feat, sigma_init=0.017, bias=True)

                self.
                self.weight_p = nn.Parameter(torch.Tensor(self.num_nodes_types,  self.in_feat, self.out_feat))
                self.bias_2 = nn.Parameter(torch.Tensor(self.num_nodes_types, self.out_feat))       
                self.sigma_weight = nn.Parameter(torch.Tensor(self.num_nodes_types, self.in_feat, self.out_feat).fill_(0.017))
                self.register_buffer("epsilon_weight", torch.zeros(self.num_nodes_types, self.in_feat, self.out_feat))
                self.sigma_bias = nn.Parameter(torch.Tensor(self.num_nodes_types, self.out_feat).fill_(sigma_init))
                self.register_buffer("epsilon_bias", torch.zeros(self.num_nodes_types, self.out_feat))
                std = math.sqrt(3 / 64)
                nn.init.uniform(self.weight_2, -std, std)
                nn.init.uniform(self.bias_2, -std, std)     
            """ 
            """ 

            if 'critic' in self.rl_learner_type.lower():  
                #self.q_noisy_linear = NoisyFactorizedLinear(self.in_feat, self.out_feat, sigma_zero=0.4, bias=True)
                self.critic_noisy_linear = NoisyLinear(self.in_feat, self.critic_out_feat, sigma_init=0.017, bias=True)                    
            if 'actor' in self.rl_learner_type.lower(): 
                #self.q_noisy_linear = NoisyFactorizedLinear(self.in_feat, self.out_feat, sigma_zero=0.4, bias=True)
                self.actor_noisy_linear = NoisyLinear(self.in_feat, self.actor_out_feat, sigma_init=0.017, bias=True)     
            """ 
            if self.rl_learner_type == "Q_Learning":                
                #UNIQUE Q_LEARNER LAYER 

                # weight bases in equation (3)

                self.weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                        self.out_feat))
                nn.init.xavier_uniform_(self.weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_nodes_types:
                    # linear combination coefficients in equation (3)
                    self.w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.w_comp_input,
                                                gain=nn.init.calculate_gain('relu')) 
                    
                if self.bias:
                    self.bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.out_feat)) 
                    nn.init.uniform_(self.bias_prediction_input) 

                if self.noisy:
                    self.sigma_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.in_feat,
                                                                                   self.out_feat).fill_(0.017))
                    self.register_buffer("epsilon_weight_prediction_input", torch.zeros(self.num_nodes_types, self.in_feat, self.out_feat))
                    self.sigma_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                 self.out_feat).fill_(0.017))
                    std = math.sqrt(3 / self.in_feat)
                    #nn.init.uniform(self.weight_prediction_input, -std, std)
                    #nn.init.uniform(self.bias_prediction_input, -std, std)
                    if self.bias:
                        self.register_buffer("epsilon_bias_prediction_input", torch.zeros(self.num_nodes_types, self.out_feat))
                        #nn.init.uniform(self.sigma_bias_prediction_input, -std, std)
                    


            if 'critic' in self.rl_learner_type.lower():            
                #UNIQUE CRITIC LAYER 
                # weight bases in equation (3)

                self.critic_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                        self.critic_out_feat))
                nn.init.xavier_uniform_(self.critic_weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_nodes_types:
                    # linear combination coefficients in equation (3)
                    self.critic_w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.critic_w_comp_input,
                                                gain=nn.init.calculate_gain('relu'))   
                if self.bias:
                    self.critic_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.critic_out_feat))       
                    nn.init.uniform_(self.critic_bias_prediction_input)   

                if self.noisy:
                    self.critic_sigma_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.in_feat,
                                                                                   self.critic_out_feat).fill_(0.017))
                    self.register_buffer("critic_epsilon_weight_prediction_input", torch.zeros(self.num_nodes_types, self.in_feat,
                                                                                               self.critic_out_feat))
                    self.critic_sigma_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                 self.critic_out_feat).fill_(0.017))
                    std = math.sqrt(3 / self.in_feat)
                    #nn.init.uniform(self.critic_weight_prediction_input, -std, std)
                    #nn.init.uniform(self.critic_bias_prediction_input, -std, std)
                    if self.bias:
                        self.register_buffer("critic_epsilon_bias_prediction_input", torch.zeros(self.num_nodes_types, self.critic_out_feat))
                        #nn.init.uniform(self.critic_sigma_bias_prediction_input, -std, std)
                    


            if 'actor' in self.rl_learner_type.lower(): 
                #UNIQUE actor LAYER 
                # weight bases in equation (3)

                self.actor_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                        self.actor_out_feat))
                nn.init.xavier_uniform_(self.actor_weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_nodes_types:
                    # linear combination coefficients in equation (3)
                    self.actor_w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.actor_w_comp_input,
                                                gain=nn.init.calculate_gain('relu'))  
                if self.bias:
                    self.actor_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.actor_out_feat))               
                    nn.init.uniform_(self.actor_bias_prediction_input)   

                if self.noisy:
                    self.actor_sigma_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.in_feat,
                                                                                   self.actor_out_feat).fill_(0.017))
                    self.register_buffer("actor_epsilon_weight_prediction_input", torch.zeros(self.num_nodes_types, self.in_feat,
                                                                                              self.actor_out_feat))
                    self.actor_sigma_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                 self.actor_out_feat).fill_(0.017))
                    std = math.sqrt(3 / self.in_feat)
                    #nn.init.uniform(self.actor_weight_prediction_input, -std, std)
                    #nn.init.uniform(self.actor_bias_prediction_input, -std, std)
                    if self.bias:
                        self.register_buffer("actor_epsilon_bias_prediction_input", torch.zeros(self.num_nodes_types, self.actor_out_feat))
                        #nn.init.uniform(self.actor_sigma_bias_prediction_input, -std, std)


        else:
            if self.rl_learner_type == "Q_Learning":      

                #Q LEARNER INPUT LAYER

                self.weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                        self.hidden_layers_size))
                nn.init.xavier_uniform_(self.weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_nodes_types:
                    # linear combination coefficients in equation (3)
                    self.w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.w_comp_input,
                                                gain=nn.init.calculate_gain('relu'))       
                    
                if self.bias:
                    self.bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size)) 
                    nn.init.uniform_(self.bias_prediction_input)   

                if self.noisy:
                    self.sigma_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.in_feat,
                                                                                   self.hidden_layers_size).fill_(0.017))
                    self.register_buffer("epsilon_weight_prediction_input", torch.zeros(self.num_nodes_types, self.in_feat,
                                                                                        self.hidden_layers_size))
                    std = math.sqrt(3 / self.in_feat)
                    #nn.init.uniform(self.weight_prediction_input, -std, std)
                    #nn.init.uniform(self.bias_prediction_input, -std, std)
                    if self.bias:
                        self.sigma_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                     self.hidden_layers_size).fill_(0.017))
                        self.register_buffer("epsilon_bias_prediction_input", torch.zeros(self.num_nodes_types, self.hidden_layers_size))
                        #nn.init.uniform(self.sigma_bias_prediction_input, -std, std)

                    


                #HIDDEN Q LEARNER LAYERS

                for n in range(self.n_hidden -1):
                    
                    weight_prediction = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.hidden_layers_size))
                    nn.init.xavier_uniform_(weight_prediction,
                                                gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        w_comp_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                        nn.init.xavier_uniform_(w_comp_prediction,
                                                    gain=nn.init.calculate_gain('relu')) 
                    if self.bias:
                        bias_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size)) 
                        nn.init.uniform_(bias_prediction)   

                    if self.noisy:
                        sigma_weight_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size,
                                                                                       self.hidden_layers_size).fill_(0.017))
                        self.register_buffer("epsilon_weight_prediction_" + str(n), torch.zeros(self.num_nodes_types,
                                                                                            self.hidden_layers_size,
                                                                                            self.hidden_layers_size))
                        sigma_bias_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                     self.hidden_layers_size).fill_(0.017))
                        std = math.sqrt(3 / self.hidden_layers_size)
                        #nn.init.uniform(weight_prediction, -std, std)
                        #nn.init.uniform(bias_prediction, -std, std)
                        if self.bias:
                            self.register_buffer("epsilon_bias_prediction_" + str(n),
                                                 torch.zeros(self.num_nodes_types,self.hidden_layers_size))
                            #nn.init.uniform(sigma_bias_prediction, -std, std)

                            
                            
                    self.prediction_weights.append(weight_prediction)
                    if self.num_bases < self.num_nodes_types:            
                        self.prediction_w_comps.append(w_comp_prediction)
                    if self.bias:
                        self.prediction_biases.append(bias_prediction)   
                    if self.noisy:
                        self.sigma_prediction_weights.append(sigma_weight_prediction)
                        self.epsilon_prediction_weights.append(getattr(self, "epsilon_weight_prediction_" + str(n)))
                        if self.bias:
                            self.sigma_prediction_biases.append(sigma_bias_prediction)
                            self.epsilon_prediction_biases.append(getattr(self, "epsilon_bias_prediction_" + str(n)))                   
                        


                #OUTPUT Q LEARNER LAYER 
                self.weight_prediction_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                        self.out_feat))
                nn.init.xavier_uniform_(self.weight_prediction_output,
                                            gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_nodes_types:
                    # linear combination coefficients in equation (3)
                    self.w_comp_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.w_comp_output,
                                                gain=nn.init.calculate_gain('relu'))        
                    
                if self.bias:
                    self.bias_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.out_feat)) 
                    nn.init.uniform_(self.bias_prediction_output)   
                
                if self.noisy:
                    self.sigma_weight_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size,
                                                                                   self.out_feat).fill_(0.017))
                    self.register_buffer("epsilon_weight_prediction_output", torch.zeros(self.num_nodes_types, self.hidden_layers_size,
                                                                                        self.out_feat))
                    std = math.sqrt(3 / self.hidden_layers_size)
                    #nn.init.uniform(self.weight_prediction_output, -std, std)
                    #nn.init.uniform(self.bias_prediction_output, -std, std)
                    if self.bias:
                        self.sigma_bias_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                     self.out_feat).fill_(0.017))
                        self.register_buffer("epsilon_bias_prediction_output", torch.zeros(self.num_nodes_types, self.out_feat))
                        #nn.init.uniform(self.sigma_bias_prediction_output, -std, std)


            if 'critic' in self.rl_learner_type.lower():  
        
        

                #CRITIC INPUT LAYER

                self.critic_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                        self.hidden_layers_size))
                nn.init.xavier_uniform_(self.critic_weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_nodes_types:
                    # linear combination coefficients in equation (3)
                    self.critic_w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.critic_w_comp_input,
                                                gain=nn.init.calculate_gain('relu')) 
                if self.bias:
                    self.critic_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size)) 
                    nn.init.uniform_(self.critic_bias_prediction_input)   
                    
                    
                if self.noisy:
                    self.critic_sigma_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.in_feat,
                                                                                   self.hidden_layers_size).fill_(0.017))
                    self.register_buffer("critic_epsilon_weight_prediction_input", torch.zeros(self.num_nodes_types, self.in_feat,
                                                                                        self.hidden_layers_size))
                    std = math.sqrt(3 / self.in_feat)
                    #nn.init.uniform(self.critic_weight_prediction_input, -std, std)
                    #nn.init.uniform(self.critic_bias_prediction_input, -std, std)
                    if self.bias:
                        self.critic_sigma_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                     self.hidden_layers_size).fill_(0.017))
                        self.register_buffer("critic_epsilon_bias_prediction_input", torch.zeros(self.num_nodes_types, self.hidden_layers_size))
                        #nn.init.uniform(self.critic_sigma_bias_prediction_input, -std, std)



                #HIDDEN CRITIC LAYERS

                for n in range(self.n_hidden -1):
                    
                    critic_weight_prediction = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.hidden_layers_size))
                    nn.init.xavier_uniform_(critic_weight_prediction,
                                                gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        critic_w_comp_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                        nn.init.xavier_uniform_(critic_w_comp_prediction,
                                                    gain=nn.init.calculate_gain('relu')) 

                    if self.bias:
                        critic_bias_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size)) 
                        nn.init.uniform_(critic_bias_prediction)   

                    if self.noisy:
                        critic_sigma_weight_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size,
                                                                                       self.hidden_layers_size).fill_(0.017))
                        self.register_buffer("critic_epsilon_weight_prediction_" + str(n), torch.zeros(self.num_nodes_types,
                                                                                            self.hidden_layers_size,
                                                                                            self.hidden_layers_size))
                        critic_sigma_bias_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                     self.hidden_layers_size).fill_(0.017))
                        std = math.sqrt(3 / self.hidden_layers_size)
                        #nn.init.uniform(critic_weight_prediction, -std, std)
                        #nn.init.uniform(critic_bias_prediction, -std, std)
                        if self.bias:
                            self.register_buffer("critic_epsilon_bias_prediction_" + str(n),
                                                 torch.zeros(self.num_nodes_types,self.hidden_layers_size))
                            #nn.init.uniform(critic_sigma_bias_prediction, -std, std)

                            
                    self.critic_prediction_weights.append(critic_weight_prediction)
                    if self.num_bases < self.num_nodes_types:            
                        self.critic_prediction_w_comps.append(critic_w_comp_prediction)
                    if self.bias:
                        self.critic_prediction_biases.append(critic_bias_prediction)   
                    if self.noisy:
                        self.critic_sigma_prediction_weights.append(critic_sigma_weight_prediction)
                        self.critic_epsilon_prediction_weights.append(getattr(self, "critic_epsilon_weight_prediction_" + str(n)))
                        if self.bias:
                            self.critic_sigma_prediction_biases.append(critic_sigma_bias_prediction)
                            self.critic_epsilon_prediction_biases.append(getattr(self, "critic_epsilon_bias_prediction_" + str(n)))                   
                        


                #OUTPUT CRITIC LAYER 
                self.critic_weight_prediction_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                        self.critic_out_feat))
                nn.init.xavier_uniform_(self.weight_prediction_output,
                                            gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_nodes_types:
                    # linear combination coefficients in equation (3)
                    self.critic_w_comp_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.critic_w_comp_output,
                                                gain=nn.init.calculate_gain('relu'))   
                    
                if self.bias:
                    self.critic_bias_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.critic_out_feat)) 
                    nn.init.uniform_(self.critic_bias_prediction_output)   

                if self.noisy:
                    self.critic_sigma_weight_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size,
                                                                                   self.critic_out_feat).fill_(0.017))
                    self.register_buffer("critic_epsilon_weight_prediction_output", torch.zeros(self.num_nodes_types, self.hidden_layers_size,
                                                                                        self.critic_out_feat))
                    std = math.sqrt(3 / self.hidden_layers_size)
                    #nn.init.uniform(self.critic_weight_prediction_output, -std, std)
                    #nn.init.uniform(self.critic_bias_prediction_output, -std, std)
                    if self.bias:
                        self.critic_sigma_bias_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                     self.critic_out_feat).fill_(0.017))
                        self.register_buffer("critic_epsilon_bias_prediction_output", torch.zeros(self.num_nodes_types, self.critic_out_feat))
                        #nn.init.uniform(self.critic_sigma_bias_prediction_output, -std, std)


            if 'actor' in self.rl_learner_type.lower():    
        

                #actor INPUT LAYER

                self.actor_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                        self.hidden_layers_size))
                nn.init.xavier_uniform_(self.actor_weight_prediction_input,
                                            gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_nodes_types:
                    # linear combination coefficients in equation (3)
                    self.actor_w_comp_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.actor_w_comp_input,
                                                gain=nn.init.calculate_gain('relu')) 
                    
                if self.bias:
                    self.actor_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size)) 
                    nn.init.uniform_(self.actor_bias_prediction_input)   

                if self.noisy:
                    self.actor_sigma_weight_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, self.in_feat,
                                                                                   self.hidden_layers_size).fill_(0.017))
                    self.register_buffer("actor_epsilon_weight_prediction_input", torch.zeros(self.num_nodes_types, self.in_feat,
                                                                                        self.hidden_layers_size))
                    std = math.sqrt(3 / self.in_feat)
                    #nn.init.uniform(self.actor_weight_prediction_input, -std, std)
                    #nn.init.uniform(self.actor_bias_prediction_input, -std, std)
                    if self.bias:
                        self.actor_sigma_bias_prediction_input = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                     self.hidden_layers_size).fill_(0.017))
                        self.register_buffer("actor_epsilon_bias_prediction_input", torch.zeros(self.num_nodes_types, self.hidden_layers_size))
                        #nn.init.uniform(self.actor_sigma_bias_prediction_input, -std, std)



                #HIDDEN actor LAYERS

                for n in range(self.n_hidden -1):
                    
                    actor_weight_prediction = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                            self.hidden_layers_size))
                    nn.init.xavier_uniform_(actor_weight_prediction,
                                                gain=nn.init.calculate_gain('relu'))
                    if self.num_bases < self.num_nodes_types:
                        # linear combination coefficients in equation (3)
                        actor_w_comp_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                        nn.init.xavier_uniform_(actor_w_comp_prediction,
                                                    gain=nn.init.calculate_gain('relu')) 

                    if self.bias:
                        actor_bias_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size)) 
                        nn.init.uniform_(actor_bias_prediction)   
                    if self.noisy:
                        actor_sigma_weight_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size,
                                                                                       self.hidden_layers_size).fill_(0.017))
                        self.register_buffer("actor_epsilon_weight_prediction_" + str(n), torch.zeros(self.num_nodes_types,
                                                                                            self.hidden_layers_size,
                                                                                            self.hidden_layers_size))
                        actor_sigma_bias_prediction = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                     self.hidden_layers_size).fill_(0.017))
                        std = math.sqrt(3 / self.hidden_layers_size)
                        #nn.init.uniform(actor_weight_prediction, -std, std)
                        #nn.init.uniform(actor_bias_prediction, -std, std)
                        if self.bias:
                            self.register_buffer("actor_epsilon_bias_prediction_" + str(n),
                                                 torch.zeros(self.num_nodes_types,self.hidden_layers_size))
                            #nn.init.uniform(actor_sigma_bias_prediction, -std, std)


                    self.actor_prediction_weights.append(actor_weight_prediction)
                    if self.num_bases < self.num_nodes_types:            
                        self.actor_prediction_w_comps.append(actor_w_comp_prediction)
                    if self.bias:
                        self.actor_prediction_biases.append(actor_bias_prediction)   
                    if self.noisy:
                        self.actor_sigma_prediction_weights.append(actor_sigma_weight_prediction)
                        self.actor_epsilon_prediction_weights.append(getattr(self, "actor_epsilon_weight_prediction_" + str(n)))
                        if self.bias:
                            self.actor_sigma_prediction_biases.append(actor_sigma_bias_prediction)
                            self.actor_epsilon_prediction_biases.append(getattr(self, "actor_epsilon_bias_prediction_" + str(n)))                   
                        


                #OUTPUT actor LAYER 
                self.actor_weight_prediction_output = nn.Parameter(torch.Tensor(self.num_bases, self.hidden_layers_size,
                                                        self.actor_out_feat))
                nn.init.xavier_uniform_(self.weight_prediction_output,
                                            gain=nn.init.calculate_gain('relu'))
                if self.num_bases < self.num_nodes_types:
                    # linear combination coefficients in equation (3)
                    self.actor_w_comp_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.num_bases))
                    nn.init.xavier_uniform_(self.actor_w_comp_output,
                                                gain=nn.init.calculate_gain('relu'))     
                    
                if self.bias:
                    self.actor_bias_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.actor_out_feat)) 
                    nn.init.uniform_(self.actor_bias_prediction_output)   
                    
                if self.noisy:
                    self.actor_sigma_weight_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, self.hidden_layers_size,
                                                                                   self.actor_out_feat).fill_(0.017))
                    self.register_buffer("actor_epsilon_weight_prediction_output", torch.zeros(self.num_nodes_types, self.hidden_layers_size,
                                                                                        self.actor_out_feat))
                    std = math.sqrt(3 / self.hidden_layers_size)
                    #nn.init.uniform(self.actor_weight_prediction_output, -std, std)
                    #nn.init.uniform(self.actor_bias_prediction_output, -std, std)
                    if self.bias:
                        self.actor_sigma_bias_prediction_output = nn.Parameter(torch.Tensor(self.num_nodes_types, 
                                                                                     self.actor_out_feat).fill_(0.017))
                        self.register_buffer("actor_epsilon_bias_prediction_output", torch.zeros(self.num_nodes_types, self.actor_out_feat))
                        #nn.init.uniform(self.actor_sigma_bias_prediction_output, -std, std)

                        
                        
                        
                        
                        
            
                        
    def predict(self, graph, device, testing = False):
        
        def message_func(edges):
            pass
        def reduce_func(nodes):  
            pass    
        
        
        def apply_func(nodes):
            """
            if self.noisy:
                if self.n_hidden == 0:
                    if self.rl_learner_type == "Q_Learning":    
                        pred = self.q_noisy_linear.forward(nodes.data['hid'])
                        #print("pred size", pred.size())
                    if 'critic' in self.rl_learner_type.lower():  
                        value = self.critic_noisy_linear.forward(nodes.data['hid'])   
                        #print("value size", value.size())
                    if 'actor' in self.rl_learner_type.lower(): 
                        actions = self.actor_noisy_linear.forward(nodes.data['hid'])   
                        #print("actions size", actions.size())

            """
            

            if self.n_hidden == 0:
                if self.rl_learner_type == "Q_Learning":                    
                    #UNIQUE Q LEARNER LAYER 
                    if self.num_bases < self.num_nodes_types:
                        # generate all weights from bases (equation (3))
                        weight_prediction_input = self.weight_prediction_input.view(self.in_feat, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_prediction_input = torch.matmul(self.w_comp_input, weight_prediction_input).view(self.num_nodes_types,
                                                                    self.in_feat, self.out_feat)#.to("cuda")     
                    else:
                        weight_prediction_input = self.weight_prediction_input#.to("cuda")    

                    w_prediction_input = weight_prediction_input[nodes.data['node_type']]#.to("cuda")   
                    if self.bias:
                        bias_prediction_input = self.bias_prediction_input[nodes.data['node_type']]
                    
                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.epsilon_weight_prediction_input[nodes.data['node_type']].size()).normal_()
                        #e_w = torch.randn(self.epsilon_weight_prediction_input[nodes.data['node_type']].size())
                        w_prediction_input += self.sigma_weight_prediction_input[nodes.data['node_type']] * Variable(e_w)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.epsilon_bias_prediction_input[nodes.data['node_type']].size()).normal_()
                            #e_b = torch.randn(self.epsilon_bias_prediction_input[nodes.data['node_type']].size())
                            bias_prediction_input += self.sigma_bias_prediction_input[nodes.data['node_type']] * Variable(e_b)

                    if self.use_message_module:
                        #print("hid size", nodes.data['hid'].size())
                        pred = torch.bmm(nodes.data['hid'].unsqueeze(1), w_prediction_input).squeeze()
                    else:
                        pred = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), w_prediction_input).squeeze()


                    pred = pred.view(-1, self.out_feat)

                    if self.bias:
                        pred = pred + bias_prediction_input

                    pred = pred.squeeze()


                    if self.dropout:
                        pred = torch.nn.functional.dropout(pred, p=0.5, training=True, inplace=False)

                if 'critic' in self.rl_learner_type.lower(): 
                    #UNIQUE CRITIC LAYER 
                    if self.num_bases < self.num_nodes_types:
                        # generate all weights from bases (equation (3))
                        critic_weight_prediction_input = self.critic_weight_prediction_input.view(self.in_feat, self.num_bases, self.critic_out_feat)#.to("cuda")     
                        critic_weight_prediction_input = torch.matmul(self.critic_w_comp_input, critic_weight_prediction_input).view(self.num_nodes_types,
                                                                    self.in_feat, self.critic_out_feat)#.to("cuda")     
                    else:
                        critic_weight_prediction_input = self.critic_weight_prediction_input#.to("cuda")    

                    #print('critic_weight_prediction_input size', critic_weight_prediction_input.size())
                    #print('node.data[nodetype]', nodes.data['node_type'])
                    critic_w_prediction_input = critic_weight_prediction_input[nodes.data['node_type']]#.to("cuda")   
                    critic_bias_prediction_input = self.critic_bias_prediction_input[nodes.data['node_type']]
                    
                    
                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.critic_epsilon_weight_prediction_input[nodes.data['node_type']].size()).normal_()       
                        #e_w = torch.randn(self.critic_epsilon_weight_prediction_input[nodes.data['node_type']].size())
                        critic_w_prediction_input += self.critic_sigma_weight_prediction_input[nodes.data['node_type']] * Variable(e_w).to(device)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.critic_epsilon_bias_prediction_input[nodes.data['node_type']].size()).normal_()
                            #e_b = torch.randn(self.critic_epsilon_bias_prediction_input[nodes.data['node_type']].size())
                            critic_bias_prediction_input += self.critic_sigma_bias_prediction_input[nodes.data['node_type']] * Variable(e_b).to(device)
                            
                    if self.use_message_module:
                        value = torch.bmm(nodes.data['hid'].unsqueeze(1), critic_w_prediction_input).squeeze()
                    else:
                        value = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), critic_w_prediction_input).squeeze()

                    value = value.view(-1, self.critic_out_feat)
                    if self.bias:
                        value = value + critic_bias_prediction_input

                    value = value.squeeze()


                if 'actor' in self.rl_learner_type.lower(): 
                    #UNIQUE ACTOR LAYER 
                    if self.num_bases < self.num_nodes_types:
                        # generate all weights from bases (equation (3))
                        actor_weight_prediction_input = self.actor_weight_prediction_input.view(self.in_feat, self.num_bases, self.actor_out_feat)#.to("cuda")     
                        actor_weight_prediction_input = torch.matmul(self.actor_w_comp_input, actor_weight_prediction_input).view(self.num_nodes_types,
                                                                    self.in_feat, self.actor_out_feat)#.to("cuda")     
                    else:
                        actor_weight_prediction_input = self.actor_weight_prediction_input#.to("cuda")    

                    actor_w_prediction_input = actor_weight_prediction_input[nodes.data['node_type']]#.to("cuda")   
                    actor_bias_prediction_input = self.actor_bias_prediction_input[nodes.data['node_type']]
                    
                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.actor_epsilon_weight_prediction_input[nodes.data['node_type']].size()).normal_()
                        #e_w = torch.randn(self.actor_epsilon_weight_prediction_input[nodes.data['node_type']].size())
                        actor_w_prediction_input += self.actor_sigma_weight_prediction_input[nodes.data['node_type']] * Variable(e_w).to(device)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.actor_epsilon_bias_prediction_input[nodes.data['node_type']].size()).normal_()
                            #e_b = torch.randn(self.actor_epsilon_bias_prediction_input[nodes.data['node_type']].size())
                            actor_bias_prediction_input += self.actor_sigma_bias_prediction_input[nodes.data['node_type']] * Variable(e_b).to(device)
                            

                    if self.use_message_module:
                        actions = torch.bmm(nodes.data['hid'].unsqueeze(1), actor_w_prediction_input).squeeze()
                    else:
                        actions = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), actor_w_prediction_input).squeeze()

                    actions = actions.view(-1, self.actor_out_feat)
                    if self.bias:
                        actions = actions + actor_bias_prediction_input

                    actions = actions.squeeze()       






            else:
                if self.rl_learner_type == "Q_Learning":      
                    #INPUT Q LEARNER LAYER 
                    if self.num_bases < self.num_nodes_types:
                        # generate all weights from bases (equation (3))
                        weight_prediction_input = self.weight_prediction_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        weight_prediction_input = torch.matmul(self.w_comp_input, weight_prediction_input).view(self.num_nodes_types,
                                                                    self.in_feat, self.hidden_layers_size)#.to("cuda")     
                    else:
                        weight_prediction_input = self.weight_prediction_input#.to("cuda")    

                    w_prediction_input = weight_prediction_input[nodes.data['node_type']]#.to("cuda")   
                    bias_prediction_input = self.bias_prediction_input[nodes.data['node_type']]
                    
                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.epsilon_weight_prediction_input[nodes.data['node_type']].size()).normal_()
                        #e_w = torch.randn(self.epsilon_weight_prediction_input[nodes.data['node_type']].size())
                        w_prediction_input += self.sigma_weight_prediction_input[nodes.data['node_type']] * Variable(e_w).to(device)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.epsilon_bias_prediction_input[nodes.data['node_type']].size()).normal_()
                            #e_b = torch.randn(self.epsilon_bias_prediction_input[nodes.data['node_type']].size())
                            bias_prediction_input += self.sigma_bias_prediction_input[nodes.data['node_type']] * Variable(e_b).to(device)
                                            
                        
                    if self.use_message_module:
                        pred = torch.bmm(nodes.data['hid'].unsqueeze(1), w_prediction_input).squeeze()
                    else:
                        pred = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), w_prediction_input).squeeze()


                    pred = pred.view(-1, self.hidden_layers_size)    

                    if self.bias:
                        pred = pred + bias_prediction_input

                    pred = pred.squeeze()
                    pred = self.activation(pred)   
                    if self.dropout:
                        pred = torch.nn.functional.dropout(pred, p=0.5, training=True, inplace=False)





                if 'critic' in self.rl_learner_type.lower():        
                    #INPUT CRITIC LAYER 
                    if self.num_bases < self.num_nodes_types:
                        # generate all weights from bases (equation (3))
                        critic_weight_prediction_input = self.weight_prediction_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        critic_weight_prediction_input = torch.matmul(self.critic_w_comp_input, critic_weight_prediction_input).view(self.num_nodes_types,
                                                                    self.in_feat, self.hidden_layers_size)#.to("cuda")     
                    else:
                        critic_weight_prediction_input = self.critic_weight_prediction_input#.to("cuda")    

                    critic_w_prediction_input = critic_weight_prediction_input[nodes.data['node_type']]#.to("cuda")   
                    critic_bias_prediction_input = self.critic_bias_prediction_input[nodes.data['node_type']]
                   
                
                    
                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.critic_epsilon_weight_prediction_input[nodes.data['node_type']].size()).normal_()       
                        #e_w = torch.randn(self.critic_epsilon_weight_prediction_input[nodes.data['node_type']].size())
                        critic_w_prediction_input += self.critic_sigma_weight_prediction_input[nodes.data['node_type']] * Variable(e_w).to(device)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.critic_epsilon_bias_prediction_input[nodes.data['node_type']].size()).normal_()
                            #e_b = torch.randn(self.critic_epsilon_bias_prediction_input[nodes.data['node_type']].size())
                            critic_bias_prediction_input += self.critic_sigma_bias_prediction_input[nodes.data['node_type']] * Variable(e_b).to(device)

                    if self.use_message_module:
                        value = torch.bmm(nodes.data['hid'].unsqueeze(1), critic_w_prediction_input).squeeze()
                    else:
                        value = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), critic_w_prediction_input).squeeze()
                        
                        
                        
                    value = value.view(-1, self.hidden_layers_size)    
                    if self.bias:
                        value = value + critic_bias_prediction_input
                    value = value.squeeze()
                    value = self.activation(value)   
                    if self.dropout:
                        value = torch.nn.functional.dropout(value, p=0.5, training=True, inplace=False)


                if 'actor' in self.rl_learner_type.lower(): 
                    #INPUT ACTOR LAYER 
                    if self.num_bases < self.num_nodes_types:
                        # generate all weights from bases (equation (3))
                        actor_weight_prediction_input = self.weight_prediction_input.view(self.in_feat, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                        actor_weight_prediction_input = torch.matmul(self.actor_w_comp_input, actor_weight_prediction_input).view(self.num_nodes_types,
                                                                    self.in_feat, self.hidden_layers_size)#.to("cuda")     
                    else:
                        actor_weight_prediction_input = self.actor_weight_prediction_input#.to("cuda")    

                    actor_w_prediction_input = actor_weight_prediction_input[nodes.data['node_type']]#.to("cuda")   
                    actor_bias_prediction_input = self.actor_bias_prediction_input[nodes.data['node_type']]
                    
                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.actor_epsilon_weight_prediction_input[nodes.data['node_type']].size()).normal_()
                        #e_w = torch.randn(self.actor_epsilon_weight_prediction_input[nodes.data['node_type']].size())
                        actor_w_prediction_input += self.actor_sigma_weight_prediction_input[nodes.data['node_type']] * Variable(e_w).to(device)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.actor_epsilon_bias_prediction_input[nodes.data['node_type']].size()).normal_()
                            #e_b = torch.randn(self.actor_epsilon_bias_prediction_input[nodes.data['node_type']].size())
                            actor_bias_prediction_input += self.actor_sigma_bias_prediction_input[nodes.data['node_type']] * Variable(e_b).to(device)

                    if self.use_message_module:
                        actions = torch.bmm(nodes.data['hid'].unsqueeze(1), actor_w_prediction_input).squeeze()
                    else:
                        actions = torch.bmm(torch.cat((nodes.data['hid'],nodes.data['short_current_phases']),1).unsqueeze(1), actor_w_prediction_input).squeeze()
                    actions = actions.view(-1, self.hidden_layers_size)    
                    if self.bias:
                        actions = actions + actor_bias_prediction_input
                    actions = actions.squeeze()
                    actions = self.activation(actions)   
                    if self.dropout:
                        actions = torch.nn.functional.dropout(actions, p=0.5, training=True, inplace=False)





                #HIDDEN Q LEARNER LAYERS            

                for idx in range(self.n_hidden -1):
                    if self.rl_learner_type == "Q_Learning":    
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            weight_prediction_hid = self.prediction_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            weight_prediction_hid = torch.matmul(self.prediction_w_comps[idx], weight_prediction_hid).view(self.num_nodes_types,
                                                                        self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                        else:
                            weight_prediction_hid = self.prediction_weights[idx]#.to("cuda")    
                        w_prediction_hid = weight_prediction_hid[nodes.data['node_type']]#.to("cuda")  
                        bias_prediction_hid = self.prediction_biases[idx][nodes.data['node_type']]
                        
                        if self.noisy and not testing:
                            e_w = torch.cuda.FloatTensor(self.epsilon_prediction_weights[idx][nodes.data['node_type']].size()).normal_()                            
                            #e_w = torch.randn(self.epsilon_prediction_weights[idx][nodes.data['node_type']].size())
                            w_prediction_hid += self.sigma_prediction_weights[idx][nodes.data['node_type']] * Variable(e_w).to(device)
                            if self.bias:
                                e_b = torch.cuda.FloatTensor(self.epsilon_prediction_biases[idx][nodes.data['node_type']].size()).normal_()    
                                #e_b = torch.randn(self.epsilon_prediction_biases[idx][nodes.data['node_type']].size())
                                bias_prediction_hid += self.sigma_prediction_biases[idx][nodes.data['node_type']] * Variable(e_b).to(device)
                                

                        pred = torch.bmm(pred.unsqueeze(1), w_prediction_hid).squeeze()
                        pred = pred.view(-1, self.hidden_layers_size)
                        if self.bias:
                            pred = pred + bias_prediction_hid
                        pred = pred.squeeze()
                        pred = self.activation(pred)
                        if self.dropout:
                            pred = torch.nn.functional.dropout(pred, p=0.5, training=True, inplace=False)                    



                    if 'critic' in self.rl_learner_type.lower():                                

                    #HIDDEN CRITIC LEARNER LAYERS            
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            critic_weight_prediction_hid = self.prediction_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            critic_weight_prediction_hid = torch.matmul(self.critic_prediction_w_comps[idx], critic_weight_prediction_hid).view(self.num_nodes_types,
                                                                        self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                        else:
                            critic_weight_prediction_hid = self.critic_prediction_weights[idx]#.to("cuda")    
                        critic_w_prediction_hid = critic_weight_prediction_hid[nodes.data['node_type']]#.to("cuda")  
                        critic_bias_prediction_hid = self.critic_prediction_biases[idx][nodes.data['node_type']]
                        
                        if self.noisy and not testing:
                            e_w = torch.cuda.FloatTensor(self.critic_epsilon_prediction_weights[idx][nodes.data['node_type']].size()).normal_() 
                            #e_w = torch.randn(self.critic_epsilon_prediction_weights[idx][nodes.data['node_type']].size())
                            critic_w_prediction_hid += self.critic_sigma_prediction_weights[idx][nodes.data['node_type']] * Variable(e_w).to(device)
                            if self.bias:
                                e_b = torch.cuda.FloatTensor(self.critic_epsilon_prediction_biases[idx][nodes.data['node_type']].size()).normal_()  
                                #e_b = torch.randn(self.critic_epsilon_prediction_biases[idx][nodes.data['node_type']].size())
                                critic_bias_prediction_hid += self.critic_sigma_prediction_biases[idx][nodes.data['node_type']] * Variable(e_b).to(device)
                           
                                
                        value = torch.bmm(value.unsqueeze(1), critic_w_prediction_hid).squeeze()
                        value = value.view(-1, self.hidden_layers_size)
                        if self.bias:
                            value = value + critic_bias_prediction_hid
                        value = value.squeeze()
                        value = self.activation(value)
                        if self.dropout:
                            value = torch.nn.functional.dropout(value, p=0.5, training=True, inplace=False)                               


                    if 'actor' in self.rl_learner_type.lower(): 
                    #HIDDEN ACTOR LEARNER LAYERS            
                        if self.num_bases < self.num_nodes_types:
                            # generate all weights from bases (equation (3))
                            actor_weight_prediction_hid = self.prediction_weights[idx].view(self.hidden_layers_size, self.num_bases, self.hidden_layers_size)#.to("cuda")     
                            actor_weight_prediction_hid = torch.matmul(self.actor_prediction_w_comps[idx], actor_weight_prediction_hid).view(self.num_nodes_types,
                                                                        self.hidden_layers_size, self.hidden_layers_size)#.to("cuda")     
                        else:
                            actor_weight_prediction_hid = self.actor_prediction_weights[idx]#.to("cuda")    
                        actor_w_prediction_hid = actor_weight_prediction_hid[nodes.data['node_type']]#.to("cuda")  
                        actor_bias_prediction_hid = self.actor_prediction_biases[idx][nodes.data['node_type']]
                        
                        if self.noisy and not testing:
                            e_w = torch.cuda.FloatTensor(self.actor_epsilon_prediction_weights[idx][nodes.data['node_type']].size()).normal_() 
                            #e_w = torch.randn(self.actor_epsilon_prediction_weights[idx][nodes.data['node_type']].size())
                            actor_w_prediction_hid += self.actor_sigma_prediction_weights[idx][nodes.data['node_type']] * Variable(e_w).to(device)
                            if self.bias:
                                e_b = torch.cuda.FloatTensor(self.actor_epsilon_prediction_biases[idx][nodes.data['node_type']].size()).normal_() 
                                #e_b = torch.randn(self.actor_epsilon_prediction_biases[idx][nodes.data['node_type']].size())
                                actor_bias_prediction_hid += self.actor_sigma_prediction_biases[idx][nodes.data['node_type']] * Variable(e_b).to(device)
    
                        actions = torch.bmm(actions.unsqueeze(1), actor_w_prediction_hid).squeeze()
                        actions = actions.view(-1, self.hidden_layers_size)
                        if self.bias:
                            actions = actions + actor_bias_prediction_hid
                        actions = actions.squeeze()
                        actions = self.activation(actions)
                        if self.dropout:
                            actions = torch.nn.functional.dropout(actions, p=0.5, training=True, inplace=False)                               



                if self.rl_learner_type == "Q_Learning":  

                    #OUTPUT Q LEARNER LAYER 
                    if self.num_bases < self.num_nodes_types:
                        # generate all weights from bases (equation (3))
                        weight_prediction_output = self.weight_prediction_output.view(self.hidden_layers_size, self.num_bases, self.out_feat)#.to("cuda")     
                        weight_prediction_output = torch.matmul(self.w_comp_output, weight_prediction_output).view(self.num_nodes_types,
                                                                    self.hidden_layers_size, self.out_feat)#.to("cuda")     
                    else:
                        weight_prediction_output = self.weight_prediction_output#.to("cuda")    
                    w_prediction_output = weight_prediction_output[nodes.data['node_type']]#.to("cuda")   
                    bias_prediction_output = self.bias_prediction_output[nodes.data['node_type']]
                    

                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.epsilon_weight_prediction_output[nodes.data['node_type']].size()).normal_()
                        #e_w = torch.randn(self.epsilon_weight_prediction_output[nodes.data['node_type']].size())
                        w_prediction_output += self.sigma_weight_prediction_output[nodes.data['node_type']] * Variable(e_w).to(device)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.epsilon_bias_prediction_output[nodes.data['node_type']].size()).normal_()
                            #e_b = torch.randn(self.epsilon_bias_prediction_output[nodes.data['node_type']].size())
                            bias_prediction_output += self.sigma_bias_prediction_output[nodes.data['node_type']] * Variable(e_b).to(device)

                    pred = torch.bmm(pred.unsqueeze(1), w_prediction_output).squeeze()
                    pred = pred.view(-1, self.out_feat)                
                    if self.bias:
                        pred = pred + bias_prediction_output      
                    pred = pred.squeeze()
                    if self.norm:
                        pred = pred * nodes.data['norm']




                if 'critic' in self.rl_learner_type.lower() and 'actor' in self.rl_learner_type.lower():         

                    # OUTPUT CRITIC LAYER
                    if self.num_bases < self.num_nodes_types:
                        # generate all weights from bases (equation (3))
                        critic_weight_prediction_output = self.critic_weight_prediction_output.view(self.hidden_layers_size, self.num_bases, self.critic_out_feat)#.to("cuda")     
                        critic_weight_prediction_output = torch.matmul(self.critic_w_comp_output, critic_weight_prediction_output).view(self.num_nodes_types,
                                                                    self.hidden_layers_size, self.critic_out_feat)#.to("cuda")     
                    else:
                        critic_weight_prediction_output = self.critic_weight_prediction_output#.to("cuda")    
                    critic_w_prediction_output = critic_weight_prediction_output[nodes.data['node_type']]#.to("cuda")   
                    critic_bias_prediction_output = self.critic_bias_prediction_output[nodes.data['node_type']]
                    
                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.critic_epsilon_weight_prediction_output[nodes.data['node_type']].size()).normal_()
                        #e_w = torch.randn(self.critic_epsilon_weight_prediction_output[nodes.data['node_type']].size())
                        critic_w_prediction_output += self.critic_sigma_weight_prediction_output[nodes.data['node_type']] * Variable(e_w).to(device)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.critic_epsilon_bias_prediction_output[nodes.data['node_type']].size()).normal_()
                            #e_b = torch.randn(self.critic_epsilon_bias_prediction_output[nodes.data['node_type']].size())
                            critic_bias_prediction_output += self.critic_sigma_bias_prediction_output[nodes.data['node_type']] * Variable(e_b).to(device)

                    value = torch.bmm(value.unsqueeze(1), critic_w_prediction_output).squeeze()
                    value = value.view(-1, self.critic_out_feat)                
                    if self.bias:
                        value = value + critic_bias_prediction_output      
                    value = value.squeeze()
                    if self.norm:
                        value = value * nodes.data['norm']                        


                if 'actor' in self.rl_learner_type.lower(): 
                    #OUTPUT ACTOR LAYER
                    if self.num_bases < self.num_nodes_types:
                        # generate all weights from bases (equation (3))
                        actor_weight_prediction_output = self.actor_weight_prediction_output.view(self.hidden_layers_size, self.num_bases, self.actor_out_feat)#.to("cuda")     
                        actor_weight_prediction_output = torch.matmul(self.actor_w_comp_output, actor_weight_prediction_output).view(self.num_nodes_types,
                                                                    self.hidden_layers_size, self.actor_out_feat)#.to("cuda")     
                    else:
                        actor_weight_prediction_output = self.actor_weight_prediction_output#.to("cuda")    
                    actor_w_prediction_output = actor_weight_prediction_output[nodes.data['node_type']]#.to("cuda")   
                    actor_bias_prediction_output = self.actor_bias_prediction_output[nodes.data['node_type']]
                    
                    if self.noisy and not testing:
                        e_w = torch.cuda.FloatTensor(self.actor_epsilon_weight_prediction_output[nodes.data['node_type']].size()).normal_()                        
                        #e_w = torch.randn(self.actor_epsilon_weight_prediction_output[nodes.data['node_type']].size())
                        actor_w_prediction_output += self.actor_sigma_weight_prediction_output[nodes.data['node_type']] * Variable(e_w).to(device)
                        if self.bias:
                            e_b = torch.cuda.FloatTensor(self.actor_epsilon_bias_prediction_output[nodes.data['node_type']].size()).normal_()                            
                            #e_b = torch.randn(self.actor_epsilon_bias_prediction_output[nodes.data['node_type']].size())
                            actor_bias_prediction_output += self.actor_sigma_bias_prediction_output[nodes.data['node_type']] * Variable(e_b).to(device)
                            
                    actions = torch.bmm(actions.unsqueeze(1), actor_w_prediction_output).squeeze()
                    actions = actions.view(-1, self.actor_out_feat)                
                    if self.bias:
                        actions = actions + actor_bias_prediction_output      
                    actions = actions.squeeze()
                    if self.norm:
                        actions = actions * nodes.data['norm']                         

                        

            # RETURN RESULTS            
            r = {}
            if self.rl_learner_type == "Q_Learning":
                r['pred'] =  pred
            if 'critic'in self.rl_learner_type.lower():
                #print("value size", value.squeeze().size())
                r['value'] = value.squeeze()
            if 'actor' in self.rl_learner_type.lower():
                #print("actions size", actions.size())
                r['actions_values'] = actions

            return r



        graph.update_all(message_func, reduce_func, apply_node_func = apply_func)  
        r = []
        #if self.value_model_based:
        if self.policy != 'binary':
            r.append(graph.ndata['node_type'])
        r.append(graph.ndata['hid'])


        if self.rl_learner_type == "Q_Learning":
            r.append(graph.ndata['pred'])
        if 'actor' in self.rl_learner_type.lower():
            r.append(graph.ndata['actions_values'])
        if 'critic'in self.rl_learner_type.lower():
            r.append(graph.ndata['value'])


        #print("rl_learner_type", self.rl_learner_type)   
        return r

    
        """
        if self.rl_learner_type == "Q_Learning":        
            return graph.ndata["pred"] 
        elif 'critic' in self.rl_learner_type.lower() and 'actor' in self.rl_learner_type.lower():
            return graph.ndata['actions_values'] , graph.ndata['value']    
        """
        
        
        
        
        

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        nn.init.uniform(self.weight, -std, std)
        nn.init.uniform(self.bias, -std, std)

    def forward(self, input):
        torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
        bias = self.bias
        if bias is not None:
            torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
            bias = bias + self.sigma_bias * Variable(self.epsilon_bias)
        #print("FORWARD")
        #print(" input size ", input.size())
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        torch.randn(self.epsilon_output.size(), out=self.epsilon_output)

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * Variable(eps_out.t())
        noise_v = Variable(torch.mul(eps_in, eps_out))
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)
