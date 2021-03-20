import torch.nn.functional as F
from utils.model import *
from utils.VBNN import *
from utils.DNN import *
import collections

###########################################################################################################################################



                                            # MAIN CLASSES/FUNCTIONS/METHODS


#######################################################################################################################################


def init_model(env):
    env.graph_of_interest = env.env_params.additional_params['graph_of_interest']
    env.steps_done = 0 
    env.env_params.additional_params['tb_filename'] = str(env.env_params.additional_params['mode'] + "_" + env.env_params.additional_params['tb_filename'])
    if env.env_params.additional_params["random_objectives"]:
        env.objectives = ['column', 'line', 'full']
        for tl_id in env.Agents:  
            objective = env.r.choice(env.objectives)
            env.Agents[tl_id].objective = obj

    # CREATE MODELS AND PARAMETERS 
    if not env.env_params.additional_params["GCN"]:
        env.model = DNN_IQL(resnet = env.env_params.additional_params['resnet'],
                            rl_learner_type = env.env_params.additional_params['Policy_Type'],
                            policy = env.env_params.additional_params['policy'],
                            noisy = env.env_params.additional_params['noisy'],
                            double = env.env_params.additional_params['double_DQN'],
                            target_model = env.env_params.additional_params['target_model_update_frequency'],
                            n_actions = env.env_params.additional_params["n_actions"],
                            dueling = env.env_params.additional_params['dueling_DQN'], 
                            tl_input_dims = get_input_dims(env, lane_dim = sum(env.env_params.additional_params["lane_vars"].values()) - (1 if 'length' in env.env_params.additional_params["lane_vars"] else 0)), 
                            activation = F.elu, 
                            n_workers = env.n_workers,
                            batch_size = env.env_params.additional_params["batch_size"],
                            mini_batch_size = env.env_params.additional_params["mini_batch_size"],
                            max_phase_len = env.max_phase_len)



        if env.env_params.additional_params['mode'] == 'train':            
            env.model.train()
        elif env.env_params.additional_params['mode'] == 'test':
            old_state_dict = copy.deepcopy(env.model.state_dict())       
            env.model.eval()
            

    elif env.env_params.additional_params['GCN']:
        env.model_init = False 

        #INITIALIZE MODEL 
        gaussian_mixture = env.env_params.additional_params["gaussian_mixture"] # number of hidden units
        n_gaussians = env.env_params.additional_params["n_gaussians"] # number of hidden units

        # configurations
        n_hidden = env.env_params.additional_params["n_hidden_layers"] # number of hidden units
        n_hidden_message = env.env_params.additional_params["n_hidden_message"]
        n_hidden_aggregation = env.env_params.additional_params["n_hidden_aggregation"]
        n_hidden_prediction = env.env_params.additional_params["n_hidden_prediction"]
        n_bases = -1 # use number of relations as number of bases
        n_hidden_layers = env.env_params.additional_params["n_hidden_layers"] # use 1 input layer, 1 output layer, no hidden layer
        n_epochs = env.env_params.additional_params["model_train_epochs"] # epochs to train
        lr = env.env_params.additional_params["learning_rate"] # learning rate
        l2norm = env.env_params.additional_params["l2_norm"] # L2 norm coefficient                
        num_classes = env.env_params.additional_params["n_classes"]      
        num_rels = len(list(env.original_graphs[env.graph_of_interest].norms.values())[0])
        rel_num_bases = env.env_params.additional_params["rel_num_bases"]
        num_attention_heads = env.env_params.additional_params["num_attention_heads"]

        num_nodes_network = len(env.original_graphs[env.graph_of_interest])
        env.num_nodes_network = num_nodes_network
        num_nodes = num_nodes_network * env.env_params.additional_params["batch_size"]
        num_tl_nodes_network = len(env.original_graphs[env.env_params.additional_params["graph_of_interest"]].parent_nid_)
        env.num_tl_nodes_network = num_tl_nodes_network
        num_tl_nodes = num_tl_nodes_network * env.env_params.additional_params["batch_size"]

        print(env.original_graphs[env.graph_of_interest].nodes_types.values())
        num_nodes_types = len(list(env.original_graphs[env.graph_of_interest].nodes_types.values()))
        if 'veh' in env.original_graphs[env.graph_of_interest].nodes_types.keys() and not env.env_params.additional_params["veh_as_nodes"] :
            num_nodes_types -=1

        std_attention = env.env_params.additional_params["normalize attention"]




        nodes_types_num_bases = env.env_params.additional_params["node_types_num_bases"]
        hidden_layers_size = env.env_params.additional_params["nn_layers_size"]
        n_convolutional_layers = env.env_params.additional_params["n_convolutional_layers"]




        train_idx = np.array(range(num_tl_nodes))
        env.train_idx = train_idx
        train_limit = len(train_idx) // 5
        env.train_limit = train_limit
        val_idx = train_idx[:train_limit]   
        env.val_idx = val_idx
        train_idx = train_idx[train_limit:]
        env.train_idx = train_idx

        node_embedding_size = env.env_params.additional_params["nn_layers_size"]        



        state_vars = env.env_params.additional_params["state_vars"] + ["hid"]        


        node_state_size = env.node_state_size


        value_model_based = env.env_params.additional_params["value_model_based"]
        use_attention = env.env_params.additional_params["use_attention"]
        rl_learner_type = env.env_params.additional_params["Policy_Type"]
        dueling = True if (env.env_params.additional_params['dueling_DQN'] and rl_learner_type == 'Q_Learning') else False
        separate_actor_critic = env.env_params.additional_params['separate_actor_critic']
        n_actions = env.env_params.additional_params["n_actions"]

        prediction_size = env.env_params.additional_params["prediction_size"]    
        norm = env.env_params.additional_params["norm"]
        use_aggregation_module = env.env_params.additional_params["use_aggregation_module"]    
        use_message_module = env.env_params.additional_params["use_message_module"]    
        dropout = env.env_params.additional_params["dropout"]   
        num_propagations = env.env_params.additional_params["num_propagations"]
        resnet = env.env_params.additional_params["resnet"]
        state_first_dim_only = env.env_params.additional_params["state_first_dim_only"]
        multidimensional_attention = env.env_params.additional_params["multidimensional_attention"]
        share_initial_params_between_actions = env.env_params.additional_params["share_initial_params_between_actions"]
        nonlinearity_before_aggregation = env.env_params.additional_params["nonlinearity_before_aggregation"]
        bias_before_aggregation = env.env_params.additional_params["bias_before_aggregation"]
        noisy = env.env_params.additional_params["noisy"]
        policy = env.env_params.additional_params["policy"]

    # POSSIBILITE DE CREER UNE CLASSE QUI INCLUE PLUSIEURS FRAMEWORK AVEC DES PARAMETRES DIFFERENTS POUR FAIRE UNE CONVOLUTION


        env.model = Convolutional_Message_Passing_Framework(policy, noisy, num_attention_heads, bias_before_aggregation , nonlinearity_before_aggregation, share_initial_params_between_actions, multidimensional_attention, state_first_dim_only, gaussian_mixture, n_gaussians, value_model_based = value_model_based, use_attention = use_attention, separate_actor_critic = separate_actor_critic, n_actions = n_actions, rl_learner_type = rl_learner_type, std_attention = std_attention, state_vars = state_vars, n_convolutional_layers = n_convolutional_layers, num_nodes_types = num_nodes_types, nodes_types_num_bases = nodes_types_num_bases, node_state_dim = node_state_size, node_embedding_dim = node_embedding_size, num_rels = num_rels, n_hidden_message = n_hidden_message, n_hidden_aggregation = n_hidden_aggregation, n_hidden_prediction = n_hidden_prediction, hidden_layers_size = hidden_layers_size, prediction_size = prediction_size, num_propagations = num_propagations, rel_num_bases = rel_num_bases, norm = norm, bias = True, activation = F.elu, use_message_module = use_message_module, use_aggregation_module = use_aggregation_module, dropout = dropout, resnet = resnet, dueling = dueling)

        env.model.nodes_types = env.original_graphs[env.graph_of_interest].nodes_types

        p = 0 
        for param in env.model.parameters():
            print(param.size())
            p+=np.prod(param.size())

        if value_model_based:
            env.model.value_transition_model.append(NN(input_size = node_embedding_size + 1 , output_size = node_embedding_size, hidden_size = node_embedding_size, n_hidden_layers = env.env_params.additional_params['n_hidden_value_transition_model'], bias = True, activation = F.elu, dropout = None, bn = False))

        if env.env_params.additional_params['mode'] == 'train':            
            env.model.train()
        elif env.env_params.additional_params['mode'] == 'test':
            env.model.load_state_dict(torch.load(env.env_params.additional_params['load_model_path'], map_location='cpu'))
            env.model.eval()


    if env.env_params.additional_params['mode'] == 'train':
    # optimizer
        env.model.optimizer = torch.optim.Adam(env.model.parameters(), lr= env.env_params.additional_params["learning_rate"], weight_decay=env.env_params.additional_params["l2_norm"])                 
    env.model.original_graphs  = env.original_graphs
    env.model.env_params = env.env_params
    env.model.Agents = env.Agents
    env.model.n_workers = env.n_workers
    if env.env_params.additional_params["GCN"]:
        env.model.num_nodes_network = env.num_nodes_network    

    return env



#########################################################################################################################################


                                    # SUB-FUNCTIONS


#####################################################################################################################################


def get_input_dims(env, lane_dim):
    tl_input_dims = collections.OrderedDict()
    for tl_id in env.Agents:
        n_lanes = len(env.Agents[tl_id].inb_lanes + env.Agents[tl_id].outb_lanes)
        dim = 0
        for var_name, var_dim in env.env_params.additional_params['tl_vars'].items():
            if type(var_name) is int:
                var = env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[var_name]
                dim += var_dim
        if 'time_since_last_action' in env.env_params.additional_params['tl_vars']:
            dim +=1
            
        dim+= env.Agents[tl_id].n_phases


        for link_idx,link in enumerate(env.Agents[tl_id].unordered_connections_trio):
            if 'open' in env.env_params.additional_params['connection_vars']:                    
                dim+=1
            if 'current_priority' in env.env_params.additional_params['connection_vars']:
                dim+=1
            if 'nb_switch_to_open' in env.env_params.additional_params['connection_vars']:
                dim+=1
            if 'priority_next_open' in env.env_params.additional_params['connection_vars']:
                dim+=1


            if env.env_params.additional_params['phase_state']:
                for idx,phase in enumerate(env.Agents[tl_id].phases_defs):
                    if (idx+1) <= env.env_params.additional_params['num_observed_next_phases']:
                        dim+=4

        dim+= n_lanes * lane_dim
            
        tl_input_dims[tl_id] = dim
        
    return tl_input_dims


