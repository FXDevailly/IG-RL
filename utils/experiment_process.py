"""Contains an experiment class for running simulations."""
import os 
import sys
from sumolib import checkBinary
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import collections
import logging
from interruptingcow import timeout
import datetime
import numpy as np
import time
import random
import traci
import math
import operator
from utils.Agent import Agent
from utils.Lane import Lane
from utils.Edge import Edge
from utils.gen_model import * 
import pickle
import networkx as nx
import dgl
from torch.utils.data import DataLoader
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
import shutil
import signal
from utils.viz_functions import *
from utils.env_classes import *
from utils.graph_tools import *
from utils.model import *


class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException
# Change the behavior of SIGALRM
    
signal.signal(signal.SIGALRM, timeout_handler)
    

    
    
    
    
########################################################################################################################################



                                            # MAIN CLASS/FUNCTIONS 



##########################################################################################################################################

class Experiment:

    def __init__(self, env,seed,n_workers):
        torch.manual_seed(seed)      
        random.seed(a=seed) 
        self.env = env
        self.env.n_workers = n_workers
        self.env.r = np.random.RandomState(int(random.random()))   
        self.setup(seed, n_workers)  
        
    def run(self, trips_dict, epoch, seed, num, n_workers, memory_queue,  request_end, learn_model_queue, comput_model_queue, reward_queue, baseline_reward_queue, Policy_Type, mode, greedy_reward_queue = None, tested_end = None, tested = None, convert_to_csv=False):
        
        if seed % n_workers== 0:
            print("run")
            s = time.time()
        
        # SAVE HYPERPARAMS AND INTER PROCESS COMMUNICATION PIPES/QUEUES
        self.env.trips_dict = trips_dict
        self.env.trips_to_complete = copy.deepcopy(list(trips_dict.keys()))
        self.env.steps_delays = []
        self.env.steps_queues = []
        self.env.steps_co2 = []
        self.env.num = num
        self.env.tested = tested
        self.env.tested_end = tested_end        
        self.env.seed = seed
        self.env.eps_threshold = self.env.env_params.additional_params['EPS_START']
        self.env.capture_counter = 0
        self.env.global_reward = 0
        if self.env.env_params.additional_params['mode'] == 'test' or self.env.env_params.additional_params['save_extended_training_stats']:
            self.env.global_delay = 0
            self.env.global_queues = 0
            self.env.global_co2 = 0
            self.env.nb_completed_trips = 0
        self.env.env_params.additional_params['mode'] = mode
        self.env.n_workers = n_workers
        self.env.memory_queue = memory_queue
        self.env.request_end = request_end
        self.env.learn_model_queue = learn_model_queue
        self.env.comput_model_queue = comput_model_queue
        self.env.reward_queue = reward_queue
        self.env.baseline_reward_queue = baseline_reward_queue
        self.env.greedy_reward_queue = greedy_reward_queue   
        

        # DRAW GRAPH OF INTEREST (USED BY THE GCN MODEL) AND SAVE IT 
        if self.env.env_params.additional_params['save_graph_drawing'] and ( seed % self.env.n_workers ==0 ):       
            draw_dgl_net(self.env.original_graphs[self.env.env_params.additional_params['graph_of_interest']], 'original_graph.png')
        
        # INITIALIZE REQUIRED OBJECTS FOR THE EXPERIMENT
        self.init_exp()
        if epoch == 0:
            if self.env.seed % self.env.n_workers ==0:
                self.env = init_model(self.env) # GENERATES MODEL 
                comput_model_queue.put(self.env.env_params) # SEND THE ENV/MODEL TO THE COMPUT
                if self.env.env_params.additional_params['Policy_Type'] == 'Q_Learning':
                    learn_model_queue.put(self.env.model) # SEND THE ENV/MODEL TO THE LEARN 

            if mode == 'train' or not self.env.env_params.additional_params['sequential_computation']:
                if self.env.seed % self.env.n_workers ==0:
                    comput_model_queue.put(self.env.model)
            else:
                if self.env.greedy:
                    if self.env.seed % self.env.n_workers != 0:
                        self.env = init_model(self.env)
                    self.env.request_end.send(self.env.model)
                else:
                    self.env.request_end.send('N/A')
                
        
        
        # DELETE/REPLACE/CREATE REPOSITORIES TO STORE RESULTS/PARAMS/CAPTURES
        self.clean_repos()

        if self.env.seed % self.env.n_workers ==0:
            print("run 1", time.time() -s, " sec")
        if self.env.env_params.additional_params['mode'] == 'train' :
            for s in range(self.env.env_params.additional_params['nb_steps_per_exp']):
                self.step()
                if self.env.seed == 0 and ( self.env.step_counter +1 ) % self.env.env_params.additional_params['nb_steps_per_exp'] == 0:
                    clear_output(wait=True)   
            self.send_results()           
        else:
            while self.env.trips_to_complete:
                self.step()
                if self.env.step_counter % 1000 == 0:
                    print('step', self.env.step_counter, 'seed :', self.env.seed, 'nb trips left :', len(self.env.trips_to_complete))

                if  self.env.step_counter % self.env.env_params.additional_params['n_avg_test'] == 0:
                    self.send_results()
                    
                # ONLY USE THIS PART FOR NY TEST 
                if self.env.env_params.additional_params['real_net_address'] == 'Manhattan.net.xml':
                    if self.env.step_counter >= int(self.env.env_params.additional_params['nb_steps_per_exp']):
                        break
                    
                    
            self.env.tested_end.send('Done')   
            self.env.request_end.send('Done') 


        
            self.env.steps_delays = np.asarray(self.env.steps_delays)
            self.env.steps_queues = np.asarray(self.env.steps_queues)
            self.env.orig_steps_delays = copy.deepcopy(self.env.steps_delays) 
            self.env.orig_steps_queues = copy.deepcopy(self.env.steps_queues)



            for veh_id in self.env.trips_dict:
                if 'truly_finished' not in self.env.trips_dict[veh_id]:
                    self.env.trips_dict[veh_id]['finish_time'] = self.env.traci_simulation_time  
                    self.env.trips_dict[veh_id]['trip_duration'] = (self.env.trips_dict[veh_id]['finish_time'] - int(float(self.env.trips_dict[veh_id]['depart'])))
                    self.env.trips_dict[veh_id]['delay'] = self.env.trips_dict[veh_id]['trip_duration']
                    self.env.trips_dict[veh_id]['nb_sec_stop'] = self.env.trips_dict[veh_id]['trip_duration']
                    self.env.trips_dict[veh_id]['CO2_emission'] = 0.0
                    self.env.trips_dict[veh_id]['truly_finished'] = False

                    
                # ADD DELAYS AND QUEUES WHEN VEHICLE WAS WAITING OUTSIDE THE NETWORK
                if 'start_time' not in self.env.trips_dict[veh_id]:
                    self.env.trips_dict[veh_id]['start_time'] = 'N/A'
                    self.env.steps_delays[int(float(self.env.trips_dict[veh_id]['depart'])):] +=float(1)
                    self.env.steps_queues[int(float(self.env.trips_dict[veh_id]['depart'])):] +=float(1)                    
                else:
                    self.env.steps_delays[int(float(self.env.trips_dict[veh_id]['depart'])): int(float(self.env.trips_dict[veh_id]['start_time']))] +=float(1)
                    self.env.steps_queues[int(float(self.env.trips_dict[veh_id]['depart'])): int(float(self.env.trips_dict[veh_id]['start_time']))] +=float(1)

            self.env.trips_dict['duration'] = self.env.traci_simulation_time
            os.system("mkdir " + self.env.env_params.additional_params["file"] + 'tensorboard' + '/' + 'network_' + str(self.env.num) + '/' +  ' > /dev/null 2>&1')
            pickfilepath = self.env.env_params.additional_params["file"] + 'tensorboard' + '/' + 'network_' + str(self.env.num) + '/' + 'traffic_' + str(self.env.seed) + '_'
            outfile = open(pickfilepath+  'trips_dict.pkl','wb')
            pickle.dump(self.env.trips_dict,outfile)
            outfile.close()             
            outfile = open(pickfilepath+  'delays.pkl','wb')
            pickle.dump(self.env.steps_delays,outfile)
            outfile.close()    
            outfile = open(pickfilepath+  'queues.pkl','wb')
            pickle.dump(self.env.steps_queues,outfile)
            outfile.close()   
            outfile = open(pickfilepath+  'co2.pkl','wb')
            pickle.dump(self.env.steps_co2,outfile)
            outfile.close()             
            outfile = open(pickfilepath+  'orig_delays.pkl','wb')
            pickle.dump(self.env.orig_steps_delays,outfile)
            outfile.close()    
            outfile = open(pickfilepath+  'orig_queues.pkl','wb')
            pickle.dump(self.env.orig_steps_queues,outfile)
            outfile.close()   

        self.env.traci_connection.close(False)
        return self.env.step_counter

    
    
    
    
    
    def setup(self,seed, n_workers):
        if self.env.env_params.additional_params['print_time_gating']:
            s = time.time()
            if seed % self.env.n_workers ==0 :
                print("0")
                print(time.time() -s, " sec")
                s = time.time()
        self.setup_state_variables()  # EXTEND THE DICTIONNARIES OF NODE VARIABLES WITH VARIABLES REPRESENTING AN ARBITRARY NUMBER OF SUB-ENTITIES 
        if self.env.env_params.additional_params['print_time_gating']:
            if seed % self.env.n_workers ==0 :
                print("1")
                print(time.time() -s, " sec")
                s = time.time()
        self.env.number_of_vehicles = 0 # USED TO CREATE INDEXES TO INDENTIFY VEHICLES
        self.setup_classes_and_networks() 
        if self.env.env_params.additional_params['print_time_gating']:
            if seed % self.env.n_workers ==0 :
                print("2")
                print(time.time() -s, " sec")
                s = time.time()
        if not self.env.env_params.additional_params['gen_trips_before_exp'] or self.env.env_params.additional_params['mode'] == 'train':
            self.setup_paths() # CREATE REALISTIC ROUTES FOR VEHICLES IN THE NETWORK
        if self.env.env_params.additional_params['print_time_gating']:
            if seed % self.env.n_workers ==0 :
                print("3")
                print(time.time() -s, " sec")
                s = time.time()

    ############################################################## EXPERIMENT     


    def init_exp(self):
        
        if self.env.env_params.additional_params['save_render'] :
            self.set_rendering_path()
        self.init_variables()
        self.init_sumo_subscriptions_and_class_variables()
        


    def step(self):# INITIALIZE A NEW STEP 
        if self.env.seed % self.env.n_workers ==0:
            s = time.time()
        # RESET  
        self.reset_agents()
        self.reset_lanes()
        self.reset_metrics()
        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed % self.env.n_workers ==0 :
                print("S1", time.time() -s ," sec")
                s = time.time()

        self.create_current_graphs() # COPIES THE ORIGINAL GRAPHS IN ORDER TO CREATE GRAPHS SPECIFIC TO THE CURRENT TIMESTEP TO WHICH WE WILL ADD VEHICLES
        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed % self.env.n_workers == 0 :
                print("S2", time.time() -s ," sec")
                s = time.time()
        if not self.env.env_params.additional_params['gen_trips_before_exp']:
            self.gen_vehicles() # GENERATE CURRENT TRAFFIC/VEHICLES
        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed % self.env.n_workers == 0 :            
                print("S3", time.time() -s ," sec")            
                s = time.time()
        self.add_vehicles() # UPDATE STATE/GRAPHS BY ADDING GENERATED VEHICLES
        
        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed % self.env.n_workers == 0 :
                print("S4", time.time() -s ," sec")        
                s = time.time()

        # INITIALIZE NODE REPRESENTATIONS
        self.init_nodes_reps()
        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed % self.env.n_workers == 0 :
                print("S5", time.time() -s ," sec")
                s = time.time()
        self.update_veh_state_and_reward()
        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed % self.env.n_workers == 0 :
                print("S6", time.time() -s ," sec")        
                s = time.time()
        if self.env.env_params.additional_params['GCN'] :                  
            self.update_lanes_nodes_states()
            self.update_tl_connection_phase_nodes_states()
            # GET STATES FOR ALL INDEPENDANT DNNs 
        elif not self.env.env_params.additional_params['GCN']:
            self.update_state_DNN()            

        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed % self.env.n_workers == 0 :
                print("S7", time.time() -s ," sec")            
                s = time.time()         


        # ACCUMULATE CURRENT REWARD ON EXPERIENCE REWARD

        self.update_metrics() # reward, queue, co2, etc...
        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed  % self.env.n_workers == 0  :
                print("S8", time.time() -s ," sec")        
                s = time.time()
        self.update_agents() #UPDATE TIME SINCE LAST ACTION AND INFO REQUIRED FOR BENCHMARK POLICIES
        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed % self.env.n_workers == 0  :
                print("S9", time.time() -s ," sec")        
                s = time.time()
        # PRINT VIZ FOR CURRENT STATE OF NETWORK ON SIM
        if self.env.env_params.additional_params['print_graph_state'] and self.env.seed  % self.env.n_workers == 0  and (self.env.step_counter +1) % 100 ==0 : 
            self.print_graph_state()

        # SELECT ACTIONS AND UPDATE SIMULATOR / SUMO
        _ = self.env.step(self.rl_actions())
        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed % self.env.n_workers == 0  :
                print("S10", time.time() -s ," sec")
                s = time.time()


            
        #SEND A BATCH TO TRAINER/EVALUATOR
        if self.env.env_params.additional_params["Policy_Type"] == 'Q_Learning' and self.env.env_params.additional_params["mode"] == 'train':
            if (self.env.step_counter - self.env.env_params.additional_params["wait_n_steps"]) % self.env.env_params.additional_params["num_steps_per_batch"] == 0  and self.env.step_counter >= (self.env.env_params.additional_params["wait_n_steps"] + self.env.env_params.additional_params["num_steps_per_batch"]):                
                self.send_experience_batch()


        self.env.step_counter +=1


        
        
#########################################################################################################################################3       
        
        
                                    # SUB-METHODS
        

        
##########################################################################################################################################      
        
        
        
        
    def reset_agents(self):
        self.env.ignored_vehicles = collections.OrderedDict()  

        for tl_id in self.env.Agents:
            self.env.Agents[tl_id].nb_stop_inb = 0
            self.env.Agents[tl_id].nb_mov_inb = 0       
            if self.env.Agents[tl_id].current_phase_idx != self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[traci.constants.TL_CURRENT_PHASE]:
                self.env.Agents[tl_id].time_since_last_action = -1
                self.env.Agents[tl_id].current_phase_idx = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[traci.constants.TL_CURRENT_PHASE]
                self.env.Agents[tl_id].current_phase = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[traci.constants.TL_RED_YELLOW_GREEN_STATE] 


            if self.env.env_params.additional_params["graph_of_interest"] == "tl_connection_lane_graph" or self.env.env_params.additional_params["graph_of_interest"] == 'full_graph':

                while self.env.Agents[tl_id].phases_defs[-1].state != self.env.Agents[tl_id].current_phase:
                    self.env.Agents[tl_id].phases_defs.insert(0,self.env.Agents[tl_id].phases_defs.pop())

            if self.env.env_params.additional_params["mode"] == 'train':
                self.env.traci_connection.trafficlight.setPhaseDuration(tl_id, 10000)            
            else:
                if type(self.env.tested) == str:
                    if 'classic' not in self.env.tested:
                        self.env.traci_connection.trafficlight.setPhaseDuration(tl_id, 10000)
                        
                    if 'strong' in self.env.tested:
                        for phase_idx, phase in enumerate(self.env.Agents[tl_id].orig_phases_defs):
                            self.env.Agents[tl_id].phases_scores[phase_idx] = 0
                            

    def create_current_graphs(self):
        self.env.current_graphs = {}
        for graph_name, graph in self.env.original_graphs.items():
            if graph_name == self.env.env_params.additional_params['graph_of_interest']:
                self.env.current_graphs[graph_name] = copy.deepcopy(graph)


    def reset_lanes(self):
        for lane_id in self.env.Lanes:
            self.env.Lanes[lane_id].reset()
        self.env.busy_lanes = set()    
        
        
        
    def gen_pedestrians(self):
        if (self.env.step_counter % (self.env.env_params.additional_params["demand_duration"])) == 0 or self.env.step_counter == 1:
            self.update_gen_probs() # PERIODICALLY UPDATES THE PROBABILITIES OF TRAFFIC GENERATION TO ENSURE VARIABILITY IN DEMAND 

        for i in range(self.env.env_params.additional_params["N_PEDESTRIAN_SAMPLES"]):
            sample = self.env.r.uniform()
            if sample <= (self.env.env_params.additional_params["PROB_PEDESTRIAN"] *(1+self.env.demand_adjuster)):
                self.env.number_of_pedestrians+=1
                ped_id = str("ped_" + str(self.env.number_of_pedestrians))
                
                # RANDOMLY SAMPLE TRAJECTORY UNTIL WE GET A VALID ONE 
                valid = False
                while not valid:
                    entering_edge = self.env.r.choice(list(self.env.entering_edges), p = self.env.new_entering_edges_probs)
                    leaving_edge = self.env.r.choice(list(self.env.leaving_edges), p = self.env.new_leaving_edges_probs)
                    if self.env.env_params.additional_params["grid"]:
                        if entering_edge[entering_edge.find('/')-1:entering_edge.find('/')+2] != leaving_edge[leaving_edge.find('/')-1:leaving_edge.find('/')+2]:
                            valid = True
                    else:
                        if entering_edge != str('-' + leaving_edge) and leaving_edge != str('-' + entering_edge):
                            valid = True
                        
                    # IF NOT VALID WE CHANGE THE PROBS (CANNOT GET OUT OF THE SOFTMAX LOOP OTHERWISE)
                    if not valid:
                        self.env.entering_adjuster = np.absolute(self.env.r.normal(loc=0, scale = self.env.env_params.additional_params["lane_demand_variance"], size=len(self.env.entering_edges)))
                        self.env.leaving_adjuster = np.absolute(self.env.r.normal(loc=0, scale = self.env.env_params.additional_params["lane_demand_variance"], size=len(self.env.leaving_edges)))
                        self.env.new_entering_edges_probs =  self.env.entering_edges_probs * (1+self.env.entering_adjuster) 
                        self.env.new_entering_edges_probs = self.env.new_entering_edges_probs -  self.env.new_entering_edges_probs.max()
                        self.env.new_leaving_edges_probs = self.env.leaving_edges_probs * (1+self.env.leaving_adjuster)
                        self.env.new_leaving_edges_probs = self.env.new_leaving_edges_probs - self.env.new_leaving_edges_probs.max()
                        self.env.new_entering_edges_probs = np.exp(self.env.new_entering_edges_probs)/np.exp(self.env.new_entering_edges_probs).sum()
                        self.env.new_leaving_edges_probs = np.exp(self.env.new_leaving_edges_probs)/np.exp(self.env.new_leaving_edges_probs).sum()                            

                        
                self.env.traci_connection.add(ped_id, entering_edge, pos = random.random() * 0, depart=-3, typeID='DEFAULT_PEDTYPE')
                rerouteTraveltime(self, ped_id)
                trip_name = str("route_" + entering_edge + "_" + leaving_edge)        
                self.env.traci_connection.appendWalkingStage(ped_id, edges, arrivalPos, duration=-1, speed=-1, stopID='')     
                
                
                
    def gen_vehicles(self):

        if (self.env.step_counter % (self.env.env_params.additional_params["demand_duration"])) == 0 or self.env.step_counter == 1:
            self.update_gen_probs() # PERIODICALLY UPDATES THE PROBABILITIES OF TRAFFIC GENERATION TO ENSURE VARIABILITY IN DEMAND 

        for idx,i in enumerate(range(self.env.env_params.additional_params["N_VEH_SAMPLES"])):
            sample = random.uniform(0,1)

            if sample <= (self.env.env_params.additional_params["PROB_VEH"] *(1+self.env.demand_adjuster)):
                self.env.number_of_vehicles+=1
                veh_id = str("veh_" + str(self.env.number_of_vehicles))

                # RANDOMLY SAMPLE TRAJECTORY UNTIL WE GET A VALID ONE 
                valid = False
                while True:
                    try:
                        entering_edge = random.choices(list(self.env.entering_edges), weights = self.env.new_entering_edges_probs)[0]
                        leaving_edge = random.choices(list(self.env.leaving_edges), weights = self.env.new_leaving_edges_probs)[0]
                        trip_name = str("route_" + entering_edge + "_" + leaving_edge)
                        self.env.traci_connection.vehicle.add(vehID = veh_id, routeID = str(trip_name + "_" + str(random.choice(range(len(self.env.shortest_paths[trip_name]))))), departSpeed=str(random.uniform(1,15)))
                        break
                    
                    except:    
                        self.env.entering_adjuster = np.absolute([random.gauss(0, self.env.env_params.additional_params["lane_demand_variance"]) for i in range(len(self.env.entering_edges))])
                        self.env.leaving_adjuster = np.absolute([random.gauss(0, self.env.env_params.additional_params["lane_demand_variance"]) for i in range(len(self.env.leaving_edges))])
                        self.env.new_entering_edges_probs =  self.env.entering_edges_probs * (1+self.env.entering_adjuster) 
                        self.env.new_entering_edges_probs = self.env.new_entering_edges_probs -  self.env.new_entering_edges_probs.max()
                        self.env.new_leaving_edges_probs = self.env.leaving_edges_probs * (1+self.env.leaving_adjuster)
                        self.env.new_leaving_edges_probs = self.env.new_leaving_edges_probs - self.env.new_leaving_edges_probs.max()
                        self.env.new_entering_edges_probs = np.exp(self.env.new_entering_edges_probs)/np.exp(self.env.new_entering_edges_probs).sum()
                        self.env.new_leaving_edges_probs = np.exp(self.env.new_leaving_edges_probs)/np.exp(self.env.new_leaving_edges_probs).sum()                            

                self.env.traci_connection.vehicle.subscribe(veh_id, [traci.constants.VAR_LANE_ID, traci.constants.VAR_NEXT_TLS,
                    traci.constants.VAR_LANE_INDEX, traci.constants.VAR_LANEPOSITION, traci.constants.VAR_ROAD_ID, traci.constants.VAR_MAXSPEED,
                    traci.constants.VAR_SPEED, traci.constants.VAR_EDGES, traci.constants.VAR_POSITION, traci.constants.VAR_SIGNALS, traci.constants.VAR_CO2EMISSION])
                self.env.traci_connection.vehicle.subscribeLeader(veh_id, 2000)




    def update_gen_probs(self):

        self.env.demand_adjuster = np.absolute(random.gauss(0, self.env.env_params.additional_params["demand_variance"]))

        self.env.entering_adjuster = np.absolute([random.gauss(0, self.env.env_params.additional_params["lane_demand_variance"]) for i in range(len(self.env.entering_edges))])        

        self.env.leaving_adjuster = np.absolute([random.gauss(0, self.env.env_params.additional_params["lane_demand_variance"]) for i in range(len(self.env.leaving_edges))])

        self.env.new_entering_edges_probs =  self.env.entering_edges_probs * (1+self.env.entering_adjuster) 
        self.env.new_entering_edges_probs = self.env.new_entering_edges_probs -  self.env.new_entering_edges_probs.max()
        self.env.new_leaving_edges_probs = self.env.leaving_edges_probs * (1+self.env.leaving_adjuster)
        self.env.new_leaving_edges_probs = self.env.new_leaving_edges_probs - self.env.new_leaving_edges_probs.max()
        self.env.new_entering_edges_probs = np.exp(self.env.new_entering_edges_probs)/np.exp(self.env.new_entering_edges_probs).sum()
        self.env.new_leaving_edges_probs = np.exp(self.env.new_leaving_edges_probs)/np.exp(self.env.new_leaving_edges_probs).sum()
        
    def add_vehicles(self):
        self.env.traci_simulation_time = self.env.traci_connection.simulation.getTime()
        if self.env.env_params.additional_params['gen_trips_before_exp']:
            for veh_id in traci.simulation.getDepartedIDList():
                if self.env.traci_connection.vehicle.isRouteValid(veh_id):    
                    if self.env.env_params.additional_params['mode'] == 'test':
                        if veh_id in self.env.trips_dict:    
                            self.env.trips_dict[veh_id]['start_time'] = self.env.traci_simulation_time -1
                            self.env.trips_dict[veh_id]['delay'] = self.env.traci_simulation_time - int(float(self.env.trips_dict[veh_id]['depart']))
                            self.env.trips_dict[veh_id]['nb_sec_stop'] = self.env.traci_simulation_time - int(float(self.env.trips_dict[veh_id]['depart']))
                            self.env.trips_dict[veh_id]['CO2_emission'] = 0
                    self.env.traci_connection.vehicle.subscribe(veh_id, [traci.constants.VAR_LANE_ID, traci.constants.VAR_NEXT_TLS,
                        traci.constants.VAR_LANE_INDEX, traci.constants.VAR_LANEPOSITION, traci.constants.VAR_ROAD_ID, traci.constants.VAR_MAXSPEED,
                        traci.constants.VAR_SPEED, traci.constants.VAR_EDGES, traci.constants.VAR_POSITION, traci.constants.VAR_SIGNALS, traci.constants.VAR_CO2EMISSION])
                    self.env.traci_connection.vehicle.subscribeLeader(veh_id, 2000)                    
                else:
                    print('veh_id :', veh_id, "INVALID")
                    self.env.self.env.traci_connection.vehicle.remove(veh_id)
                    self.env.trips_to_complete.remove(veh_id)
                    del self.env.trips_dict[veh_id]
                    
                    
        for veh_id in self.env.traci_connection.vehicle.getIDList():
            lane_id = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[traci.constants.VAR_LANE_ID]
            self.env.ignored_vehicles[veh_id] = False 
            if lane_id in self.env.lanes :
                self.env.last_lane[veh_id]= lane_id                 
            elif self.env.env_params.additional_params['ignore_central_vehicles']:
                self.env.ignored_vehicles[veh_id] = True

        if self.env.env_params.additional_params["GCN"]:        
            if self.env.env_params.additional_params["veh_as_nodes"]:
                for idx, (graph_name, graph) in enumerate(self.env.current_graphs.items()):
                    if graph_name == self.env.env_params.additional_params['graph_of_interest']:
                        count = 0
                        if "lane" in graph_name:
                            src = []
                            dst = []
                            tp = []
                            norm = []
                            node_type = []
                            for veh_id in self.env.traci_connection.vehicle.getIDList():


                                lane_id = self.env.last_lane[veh_id]

                                if veh_id not in graph.adresses_in_graph and lane_id in graph.adresses_in_graph:

                                    #REGISTER CAR IN GRAPHS
                                    veh_graph_id = graph.number_of_nodes()+count
                                    current_length = len(graph.norms[lane_id])
                                    graph.adresses_in_graph[veh_id] = veh_graph_id
                                    graph.norms[veh_id] = [0]*current_length
                                    graph.adresses_in_sumo[str(veh_graph_id)] = veh_id
                                    node_type.append(len(graph.nodes_types))

                                    #ADD SELF LOOP WITH TYPE AT THE END 
                                    src.append(graph.adresses_in_graph[veh_id])
                                    dst.append(graph.adresses_in_graph[veh_id])
                                    graph.norms[veh_id][-1] += 1     
                                    tp.append(len(graph.norms[veh_id])-1)   
                                    #CAR TO LANE - EDGE
                                    src.append(int(veh_graph_id))
                                    dst.append(int(graph.adresses_in_graph[lane_id]))
                                    #LANE TO CAR - EDGE
                                    src.append(int(graph.adresses_in_graph[lane_id]))
                                    dst.append(int(veh_graph_id))
                                    graph.norms[lane_id][-3] +=1
                                    tp.append(int(current_length-3))
                                    graph.norms[veh_id][-2] +=1
                                    tp.append(int(current_length-2))


                                    count +=1         



                            if count>0:
                                for destination, t in zip(dst,tp):
                                    norm.append([(1/(graph.norms[graph.adresses_in_sumo[str(destination)]][t]))])            
                                self.env.number_of_cars = count
                                src = torch.LongTensor(src)
                                dst = torch.LongTensor(dst)
                                edge_type = torch.LongTensor(tp)
                                edge_norm = torch.FloatTensor(norm)

                                graph.add_nodes(self.env.number_of_cars)
                                graph.add_edges(src,dst) #, {'rel_type':edge_type, 'norm':edge_norm})
                                graph.edata['rel_type'] = torch.cat((graph.edata['rel_type'][:-len(src)],torch.LongTensor(edge_type).squeeze()),0).squeeze()
                                graph.edata['norm'] = torch.cat((graph.edata['norm'][:-len(src)],torch.FloatTensor(edge_norm).squeeze()),0).squeeze()    

                                graph.ndata['node_type'] = torch.cat((graph.ndata['node_type'][:-self.env.number_of_cars],torch.LongTensor(node_type)),0) 

                            new_filt = partial(filt, identifier = [graph.nodes_types['veh']])
                            graph.nodes_lists['veh'] = graph.filter_nodes(new_filt)



            if self.env.env_params.additional_params['save_graph_drawing'] and ( self.env.seed % self.env.n_workers ==0 ) and self.env.step_counter==100:       
                draw_dgl_net(self.env.current_graphs[self.env.env_params.additional_params['graph_of_interest']], 'current_graph.png')




    def init_nodes_reps(self):
        if self.env.env_params.additional_params["GCN"]:         

            for graph_name, graph in self.env.current_graphs.items():
                if graph_name == self.env.env_params.additional_params['graph_of_interest']:
                    graph.ndata.update({'state' : torch.zeros(graph.number_of_nodes(),self.env.node_state_size, dtype = torch.float32)})



    def update_veh_state_and_reward(self):
        # UPDATE REWARD AND LANE DATA
        
        s = time.time()
        for veh_id in self.env.traci_connection.vehicle.getIDList():
            # GET RELEVANT  VEHICLE VARIABLES
            lane_id = self.env.last_lane[veh_id]
            veh_speed = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[traci.constants.VAR_SPEED]
            veh_max_speed = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[traci.constants.VAR_MAXSPEED]
            veh_position = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[traci.constants.VAR_LANEPOSITION]
            if self.env.ignored_vehicles[veh_id]:
                veh_position = veh_position + self.env.Lanes[lane_id].length
            # UPDATE REWARD PER LANE AND LANE INFORMATION
            delay, queue = self.env.Lanes[lane_id].update_lane_state_and_reward(veh_speed, veh_max_speed, veh_position, self.env.env_params.additional_params["veh_state"], self.env.ignored_vehicles[veh_id])
            self.env.busy_lanes.add(lane_id)           
            # CO2
            if self.env.env_params.additional_params['mode'] == 'test' or self.env.env_params.additional_params['save_extended_training_stats']:
                co2 = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[traci.constants.VAR_CO2EMISSION]
                self.env.co2 += co2
            if self.env.env_params.additional_params['GCN'] and self.env.env_params.additional_params['veh_as_nodes'] and veh_id in self.env.current_graphs[self.env.env_params.additional_params['graph_of_interest']].adresses_in_graph:
                self.update_veh_node_state(veh_id, lane_id)
            if self.env.env_params.additional_params['mode'] == 'test':     
                if veh_id in self.env.trips_dict:
                    self.env.trips_dict[veh_id]['delay'] += delay
                    self.env.trips_dict[veh_id]['nb_sec_stop'] += queue
                    self.env.trips_dict[veh_id]['CO2_emission'] += co2
            # UPDATE NODES REPRESENTATIONS BASED ON CURRENT STATE OF THE SIMULATION 

        self.env.busy_lanes = set(list(self.env.busy_lanes))
        if self.env.env_params.additional_params['print_time_gating']:
            if self.env.seed % self.env.n_workers == 0 :
                print("U1", time.time() -s, " sec")
        
    def update_veh_node_state(self, veh_id, lane_id):
        counter = 0           
        for var_name, var_dim in self.env.env_params.additional_params['veh_vars'].items():
            
            if type(var_name) == int:
                var = self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[var_name]
                if var_name == traci.constants.VAR_LANEPOSITION :
                    if self.env.ignored_vehicles[veh_id]:
                        var += self.env.Lanes[lane_id].length
                    if self.env.env_params.additional_params['std_lengths']:
                        var/= self.env.Lanes[lane_id].length if not self.env.env_params.additional_params['grid'] else self.env.env_params.additional_params['grid_lane_length']
                elif var_name == traci.constants.VAR_SPEED:
                    if self.env.env_params.additional_params['std_speeds']:
                        var /= self.env.env_params.additional_params['Max_Speed']
            elif var_name == 'distance_to_tl':
                var = self.env.Lanes[lane_id].length - self.env.traci_connection.vehicle.getSubscriptionResults(veh_id)[traci.constants.VAR_LANEPOSITION]
                if self.env.ignored_vehicles[veh_id]:
                    var -= self.env.Lanes[lane_id].length
                if self.env.env_params.additional_params['std_lengths']:
                    var/= self.env.env_params.additional_params['grid_lane_length'] if self.env.env_params.additional_params['grid'] else self.env.env_params.additional_params['max_lane_length']

            for graph_name, graph in self.env.current_graphs.items():
                if graph_name == self.env.env_params.additional_params['graph_of_interest']:
                    if 'lane' in graph_name:

                        # CONTINUOUS VARIABLE

                        if var_dim == 1 :
                            graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter] = var
                        # DUMMY VARIABLE
                        elif var_dim > 1: 
                            if var_name == traci.constants.VAR_SIGNALS:
                                if var == 2 or var == 10: # 2 = LEFT # 10 = LEFT+BRAKE
                                    graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + 0] = 1                                                
                                    if var == 10:
                                        graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + 2] = 1      
                                elif var == 1 or var == 9: # 1 = RIGHT # 9 = RIGHT + BRAKE
                                    graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + 1] = 1
                                    if var == 9:
                                        graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + 2] = 1
                                elif var == 8: # 8 = BRAKE 
                                    graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + 2] = 1
                            else:
                                graph.ndata['state'][graph.adresses_in_graph[veh_id]][counter + var] = 1


            counter += var_dim

        if self.env.env_params.additional_params['print_graph_state'] and self.env.seed  % self.env.n_workers == 0 :# and (self.env.step_counter +1) % 100 ==0 :
            print("veh id :", veh_id, "data :", graph.ndata['state'][graph.adresses_in_graph[veh_id]])



    def update_lanes_nodes_states(self):

        # UPDATE LANE NODE DATA
        for lane_id in self.env.lanes_in_graph:
            for graph_name, graph in self.env.current_graphs.items():
                if graph_name == self.env.env_params.additional_params['graph_of_interest']:     
                    if self.env.env_params.additional_params['lane_node_state']:  
                        if 'x' in self.env.env_params.additional_params['lane_vars']:
                            graph.ndata['state'][graph.adresses_in_graph[lane_id]][-3] = float(lane_id[-3]) #X  
                        if 'y' in self.env.env_params.additional_params['lane_vars']:                
                            graph.ndata['state'][graph.adresses_in_graph[lane_id]][-2]  = float(lane_id[-5]) #Y
                        if 'which_lane' in self.env.env_params.additional_params['lane_vars']:
                            graph.ndata['state'][graph.adresses_in_graph[lane_id]][-1] = float(lane_id[-1]) # WHICH LANE

                        if 'type' in self.env.env_params.additional_params['lane_vars']:
                            if 'bot' in lane_id:
                                graph.ndata['state'][graph.adresses_in_graph[lane_id]][-4]  = 1                      
                            if 'top' in lane_id:
                                graph.ndata['state'][graph.adresses_in_graph[lane_id]][-5]  = 1    
                            if 'left' in lane_id:
                                graph.ndata['state'][graph.adresses_in_graph[lane_id]][-6]  = 1    
                            if 'right' in lane_id:
                                graph.ndata['state'][graph.adresses_in_graph[lane_id]][-7]  = 1                                                                 
                        if 'lane' in graph_name or graph_name == 'full_graph':                           
                            for idx, (var_name, lane_state_var) in enumerate(self.env.env_params.additional_params['lane_vars'].items()):
                                if var_name == 'nb_veh':
                                    graph.ndata['state'][graph.adresses_in_graph[lane_id]][idx] = self.env.Lanes[lane_id].state[0]                                          
                                elif var_name =='avg_speed':
                                    avg_speed = self.env.Lanes[lane_id].state[1]  
                                    if self.env.env_params.additional_params['std_speeds']:
                                        avg_speed /= self.env.env_params.additional_params['Max_Speed']
                                    graph.ndata['state'][graph.adresses_in_graph[lane_id]][idx] = avg_speed
                                elif var_name =='length':
                                    graph.ndata['state'][graph.adresses_in_graph[lane_id]][idx] = self.env.Lanes[lane_id].std_length                            if self.env.env_params.additional_params['std_lengths'] else self.env.Lanes[lane_id].length 


                    if self.env.env_params.additional_params['print_graph_state'] and self.env.seed  % self.env.n_workers == 0 :# and (self.env.step_counter +1) % 100 ==0 :

                        print("lane id:", lane_id, "data :", graph.ndata['state'][graph.adresses_in_graph[lane_id]])




    def update_state_DNN(self):
        self.env.state = []
        for tl_id in self.env.Agents:
            state=[]
            for var_name, var_dim in self.env.env_params.additional_params['tl_vars'].items():
                if type(var_name) is int:
                    var = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[var_name]
                    state.append(var)
            if 'time_since_last_action' in self.env.env_params.additional_params['tl_vars']:
                state.append(self.env.Agents[tl_id].time_since_last_action)

            if 'active' in self.env.env_params.additional_params['phase_vars']:
                def_state = [0] * self.env.Agents[tl_id].n_phases         
                for idx, phase in enumerate(self.env.Agents[tl_id].orig_phases_defs):
                    if phase.state == self.env.Agents[tl_id].current_phase:
                        def_state[idx] = 1      
                state.extend(def_state)
            


            for link_idx,link in enumerate(self.env.Agents[tl_id].unordered_connections_trio):
                link_name = str(tl_id+"_link_"+str(link_idx))
                if 'open' in self.env.env_params.additional_params['connection_vars']:                    
                    if self.env.Agents[tl_id].current_phase[link_idx].lower() == 'g':   
                        state.append(1)
                    else:
                        state.append(0)
                if 'current_priority' in self.env.env_params.additional_params['connection_vars']:
                    if self.env.Agents[tl_id].current_phase[link_idx] == 'G':  
                        state.append(1)
                    else:
                        state.append(0)

                v=False
                if 'nb_switch_to_open' in self.env.env_params.additional_params['connection_vars']:
                    for idx,phase in enumerate(self.env.Agents[tl_id].phases_defs):
                        link_state = phase.state[link_idx]
                        if link_state.lower() == 'g':   
                            state.append(idx+1)
                            v = True
                            break
                if not v:
                    state.append(-1)
                            
                v=False                            
                if 'priority_next_open' in self.env.env_params.additional_params['connection_vars']:
                    for idx,phase in enumerate(self.env.Agents[tl_id].phases_defs):
                        phase = phase.state[link_idx]
                        if phase.lower() == 'g':
                            if phase == 'G':     
                                state.append(1)
                            else:
                                state.append(0)
                            v=True
                            break
                if not v:
                    state.append(-1)



                if self.env.env_params.additional_params['phase_state']:
                    for idx,phase in enumerate(self.env.Agents[tl_id].phases_defs):
                        if (idx+1) <= self.env.env_params.additional_params['num_observed_next_phases']:
                            if phase.state[link_idx].lower() == 'g':
                                state.append(1)
                            else:
                                state.append(0)
                            if phase.state[link_idx] == 'G':
                                state.append(1)
                            else:
                                state.append(0)
                            if phase.state[link_idx].lower() == 'y':
                                state.append(1)
                            else:
                                state.append(0)
                            if phase.state[link_idx].lower() == 'r':
                                state.append(1)
                            else:
                                state.append(0)


            for lane_id in self.env.Agents[tl_id].inb_lanes + self.env.Agents[tl_id].outb_lanes:

                if self.env.env_params.additional_params['std_speeds']:
                    self.env.Lanes[lane_id].state[1] /= self.env.Lanes[lane_id].max_speed
                state.extend(self.env.Lanes[lane_id].state)# 
            self.env.state.append(state)

            
        if self.env.step_counter == 0:
            self.max_len = max([len(l) for l in self.env.state])

            
        # PADDING 
        for l in self.env.state:
            l += [0]*(self.max_len - len(l))
            
    def update_tl_connection_phase_nodes_states(self):
        for tl_id in self.env.Agents:
            counter = 0
            for var_name, var_dim in self.env.env_params.additional_params['tl_vars'].items():
                if type(var_name) is int:
                    var = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[var_name]
                    for graph_name, graph in self.env.current_graphs.items():  
                        if graph_name == self.env.env_params.additional_params['graph_of_interest']:
                            if len(graph.nodes_lists['tl']) > 0 : 
                                # CONTINUOUS VARIABLE
                                if var_dim == 1 :
                                    graph.ndata['state'][graph.adresses_in_graph[tl_id]][counter] = var
                                # DUMMY VARIABLE
                                elif var_dim > 1: 
                                    graph.ndata['state'][graph.adresses_in_graph[tl_id]][counter + var] = 1


                    counter += var_dim






        for tl_idx, tl_id in enumerate(self.env.Agents):
            for graph_name, graph in self.env.current_graphs.items():  
                if graph_name == self.env.env_params.additional_params['graph_of_interest']:

                    if 'time_since_last_action' in self.env.env_params.additional_params['tl_vars']:
                        graph.ndata['state'][graph.adresses_in_graph[tl_id]][-1] = self.env.Agents[tl_id].time_since_last_action 

                    if 'x' in self.env.env_params.additional_params['tl_vars']:
                        graph.ndata['state'][graph.adresses_in_graph[tl_id]][-2] = int(tl_id[6:])%self.env.env_params.additional_params["row_num"]   
                    if 'y' in self.env.env_params.additional_params['tl_vars']:
                        graph.ndata['state'][graph.adresses_in_graph[tl_id]][-3] = int(tl_id[6:])//self.env.env_params.additional_params["col_num"]  

                        
                                    
                    if graph_name == 'full_graph':
                        if 'active' in self.env.env_params.additional_params['phase_vars']:
                            graph.ndata['state'][graph.adresses_in_graph[str(tl_id+"_phase_"+ str(self.env.Agents[tl_id].current_phase_idx))]][0] = 1
                        if 'next_yellow' in self.env.env_params.additional_params['phase_vars']:

                            for idx,phase in enumerate(self.env.Agents[tl_id].phases_defs):
                        
                                phase_name = str(tl_id+"_phase_"+str(phase.idx))
                                if 'y' not in phase.state.lower():
                                    break
                                graph.ndata['state'][graph.adresses_in_graph[phase_name]][1] = 1
             
                                    
                                
                        if self.env.env_params.additional_params['print_graph_state'] and self.env.seed  % self.env.n_workers == 0 :
                            for idx,phase in enumerate(self.env.Agents[tl_id].phases_defs):
                                phase_name = str(tl_id+"_phase_"+str(phase.idx))
                                print("phase id :",phase_name , "data :",graph.ndata['state'][graph.adresses_in_graph[phase_name]])

                    if graph_name == 'tl_connection_lane_graph' or graph_name == "full_graph" and self.env.env_params.additional_params['policy'] == 'binary':
                        for link_idx,link in enumerate(self.env.Agents[tl_id].unordered_connections_trio):
                            link_name = str(tl_id+"_link_"+str(link_idx))

                            if 'open' in self.env.env_params.additional_params['connection_vars']:
                                if self.env.Agents[tl_id].current_phase[link_idx].lower() == 'g':   
                                    graph.ndata['state'][graph.adresses_in_graph[link_name]][-1] = 1
                            if 'current_priority' in self.env.env_params.additional_params['connection_vars']:
                                if self.env.Agents[tl_id].current_phase[link_idx] == 'G':  
                                    graph.ndata['state'][graph.adresses_in_graph[link_name]][-3] = 1


                            for idx,phase in enumerate(self.env.Agents[tl_id].phases_defs):   
                                phase = phase.state[link_idx]
                                if 'nb_switch_to_open' in self.env.env_params.additional_params['connection_vars']:
                                    if phase.lower() == 'g': 
                                        graph.ndata['state'][graph.adresses_in_graph[link_name]][-2] = (idx+1)  

                                if 'priority_next_open' in self.env.env_params.additional_params['connection_vars']:
                                    if phase == 'G':     
                                        graph.ndata['state'][graph.adresses_in_graph[link_name]][-4] = 1       

                                if phase.lower() == 'g':
                                    break
                                    
                            if self.env.env_params.additional_params['phase_state']:

                                for idx,phase in enumerate(self.env.Agents[tl_id].phases_defs):
                                    if (idx+1) <= self.env.env_params.additional_params['num_observed_next_phases']:

                                        if phase.state[link_idx].lower() == 'g':
                                            graph.ndata['state'][graph.adresses_in_graph[link_name]][4*idx] = 1                
                                        if phase.state[link_idx] == 'G':
                                            graph.ndata['state'][graph.adresses_in_graph[link_name]][4*idx+1] = 1          
                                        if phase.state[link_idx].lower() == 'y':
                                            graph.ndata['state'][graph.adresses_in_graph[link_name]][4*idx+2] = 1
                                        if phase.state[link_idx].lower() == 'r':
                                            graph.ndata['state'][graph.adresses_in_graph[link_name]][4*idx+3] = 1          

                            if self.env.env_params.additional_params['print_graph_state'] and self.env.seed  % self.env.n_workers == 0 :# and (self.env.step_counter +1) % 100 ==0 :    
                                print("link name:", link_name, "data:", graph.ndata['state'][graph.adresses_in_graph[link_name]])


                    if self.env.env_params.additional_params['print_graph_state'] and self.env.seed  % self.env.n_workers == 0 :# and (self.env.step_counter +1) % 100 ==0 :
                        print("tl id :", tl_id, "data :",graph.ndata['state'][graph.adresses_in_graph[tl_id]])




    def update_agents(self):
        for tl_id in self.env.Agents:              
            self.env.Agents[tl_id].time_since_last_action +=1
            if self.env.env_params.additional_params['mode'] == 'test':
                if type(self.env.tested) == str: 
                    if 'strong_baseline' in self.env.tested :
                        if self.env.env_params.additional_params['policy'] == 'binary':
                            for lane_id in self.env.Agents[tl_id].inb_lanes:
                                self.env.Agents[tl_id].nb_stop_inb += self.env.Lanes[lane_id].nb_stop_veh
                                self.env.Agents[tl_id].nb_mov_inb += self.env.Lanes[lane_id].nb_mov_veh  
                        else:
                            for connection in self.env.Agents[tl_id].unordered_connections_trio:
                                inb_lane = connection[0]
                                connection_score= self.env.Lanes[inb_lane].nb_stop_veh * (1/self.env.Lanes[inb_lane].connection_count)
                                for phase_idx,phase in enumerate(self.env.Agents[tl_id].orig_phases_defs):
                                    link_state = phase.state[link_idx]
                                    if link_state.lower() == 'g': 
                                        self.env.Agents[tl_id].phases_scores[phase_idx] += connection_score                       


    def send_experience_batch(self):
        # Q LEARNING
        for idx, (g, labels, actions, choices, forced_mask, forced_value, actions_sizes) in enumerate(reversed(self.env.all_graphs)): # choices
            labels = np.asarray(labels) 
            if idx == 0:
                reward = labels
            elif idx < len(self.env.all_graphs):
                reward = labels + (self.env.env_params.additional_params['time_gamma'] * reward)

                # extend with FUTURE REWARD, NEXT STATE (GRAPH), IMMEDIATE REWARD t-1, NEXT_CHOICES, NEXT_FORCED_0, NEXT_FORCED_1

                self.env.all_graphs[-(idx)].extend([list(reward), self.env.all_graphs[-(idx-1)][0], self.env.all_graphs[-(idx+1)][1],self.env.all_graphs[-(idx-1)][3], self.env.all_graphs[-(idx-1)][4],self.env.all_graphs[-(idx-1)][5], self.env.all_graphs[-(idx-1)][6]])

        self.env.memory_queue.send(self.env.all_graphs[1:-1])
        
        
        self.env.all_graphs = []

    def rl_actions(self):
        activated_tls = []
        self.env.Actions = collections.OrderedDict()
        actions = [0]*self.env.original_graphs["tl_graph"].number_of_nodes()     
        action_size = [0]*self.env.original_graphs["tl_graph"].number_of_nodes()
        self.env.Targets = collections.OrderedDict()
        labels = [0]*self.env.original_graphs["tl_graph"].number_of_nodes() 
        choices = [0]*self.env.original_graphs["tl_graph"].number_of_nodes()
        forced_mask = [0]*self.env.original_graphs["tl_graph"].number_of_nodes()
        forced_value = [0]*self.env.original_graphs["tl_graph"].number_of_nodes()
        if self.env.env_params.additional_params["mode"] == 'train':
            for idx,tl_id in enumerate(self.env.Agents):
                action_size[idx]= self.env.Agents[tl_id].n_phases
                if self.env.env_params.additional_params['distance_gamma'] not in [None,0]:
                    self.env.Targets[tl_id] = self.env.Agents[tl_id].get_reward(self.env.reward_vector)
                else:
                    reward = 0
                    for lane_id in self.env.Agents[tl_id].inb_lanes:                    
                        reward += self.env.Lanes[lane_id].get_reward()
                    self.env.Targets[tl_id] = reward

                if self.env.env_params.additional_params["std_nb_veh"]:
                    n_veh = 0
                    for lane_id in self.env.Agents[tl_id].complete_controlled_lanes:
                        if 'c' not in lane_id:
                            n_veh += self.env.Lanes[lane_id].state[0]
                    self.env.Targets[tl_id] /= max(n_veh,1)

                if self.env.seed % self.env.n_workers == 0  and self.env.env_params.additional_params['print_tl_rewards']: 
                    print("tl :", tl_id, 'target reward :',self.env.Targets[tl_id])

                if self.env.env_params.additional_params['GCN']:
                    labels[list(self.env.current_graphs[self.env.graph_of_interest].parent_nid_).index(self.env.current_graphs[self.env.graph_of_interest].adresses_in_graph[tl_id])] = self.env.Targets[tl_id]
                else:
                    labels[idx] = self.env.Targets[tl_id]

        if self.env.seed % self.env.n_workers == 0 :
            print("step", self.env.step_counter)


        # Q_LEARNING
        if self.env.env_params.additional_params['Policy_Type'] == "Q_Learning": 
            self.env.request_end.send((self.env.current_graphs[self.env.graph_of_interest] if self.env.env_params.additional_params['GCN'] else self.env.state, self.env.original_graphs["tl_graph"].number_of_nodes(), torch.LongTensor(action_size).view(-1))  if not self.env.baseline else 'N/A') 

        if not self.env.baseline :

            actions = self.env.request_end.recv()
            actions = list(actions)


            if not self.env.greedy and not self.env.env_params.additional_params['gaussian_mixture'] and not self.env.env_params.additional_params['noisy']:
                if self.env.eps_threshold > self.env.env_params.additional_params['EPS_END']:
                    self.env.eps_threshold = self.env.eps_threshold * self.env.env_params.additional_params['EPS_DECAY']
                if self.env.step_counter % 50 == 0 and self.env.seed  % self.env.n_workers == 0 :
                    print("eps_threshold :", self.env.eps_threshold)

                for idx,tl_id in enumerate(self.env.Agents):
                    sample = random.random()
                    if sample < self.env.eps_threshold:
                        actions[idx] = torch.from_numpy(random.randint(0, 2)).squeeze()

        else:
            _ = self.env.request_end.recv()


        # CONSTRAINTS 

        for idx,tl_id in enumerate(self.env.Agents): 

            if self.env.Agents[tl_id].time_since_last_action+1 >= self.env.env_params.additional_params["min_time_between_actions"] and ((self.env.Agents[tl_id].time_since_last_action+1 - self.env.env_params.additional_params["min_time_between_actions"]) % self.env.env_params.additional_params["time_between_actions"] == 0 or self.env.Agents[tl_id].time_since_last_action+1 == self.env.env_params.additional_params["min_time_between_actions"]):
                
                
                if self.env.env_params.additional_params["policy"] == 'binary' and self.env.Agents[tl_id].time_since_last_action+1 >= self.env.env_params.additional_params["max_time_between_actions"]:
                    forced_mask[idx] = 1            
                    self.env.Actions[tl_id] = 1  
                    forced_value[idx] = 1
                else:                
                    if 'y' in self.env.Agents[tl_id].current_phase.lower():
                        if self.env.Agents[tl_id].time_since_last_action +1 < self.env.env_params.additional_params["yellow_duration"]:
                            forced_mask[idx] = 1
                            if self.env.env_params.additional_params["policy"] != 'binary': 
                                self.env.Actions[tl_id] = self.env.Agents[tl_id].current_phase_idx   
                                forced_value[idx] = self.env.Agents[tl_id].current_phase_idx 
                            else:
                                self.env.Actions[tl_id] = 0
                                forced_value[idx] = 0
                        else:
                            if self.env.env_params.additional_params["policy"] != 'binary': 
                                if 'y' in self.env.Agents[tl_id].phases_defs[0].state.lower():
                                    self.env.Actions[tl_id]=self.env.Agents[tl_id].current_phase_idx+1  
                                    forced_mask[idx] = 1
                                    forced_value[idx] = self.env.Agents[tl_id].current_phase_idx+1  
                                else:
                                    self.env.Actions[tl_id] = actions[idx] 
                                    forced_mask[idx] = 0
                            else:
                                self.env.Actions[tl_id] = 1
                                forced_value[idx] = 1
                                forced_mask[idx] = 1


                    else: 
                        forced_mask[idx] = 0
                        if self.env.env_params.additional_params["policy"] != 'binary': 
                            if self.env.greedy or self.env.env_params.additional_params["mode"] == 'train':
                                action = actions[idx]
                            elif type(self.env.tested) == str: 
                                if 'strong' in self.env.tested :
                                    action = max(self.env.Actions[tl_id].phases_scores.items(), key=operator.itemgetter(1))[0]
                                elif 'random' in self.env.tested:
                                    action = random.choice(list(self.env.Actions[tl_id].phases_scores.keys()))

                            if action != self.env.Agents[tl_id].current_phase_idx:
                                if 'y' in self.env.Agents[tl_id].phases_defs[0].state.lower():
                                    self.env.Actions[tl_id]=self.env.Agents[tl_id].current_phase_idx+1   
                                else:
                                    self.env.Actions[tl_id]=action
                            else:
                                self.env.Actions[tl_id]=self.env.Agents[tl_id].current_phase_idx                                  

                        else:
                            forced_mask[idx] = 0
                            if self.env.greedy or self.env.env_params.additional_params["mode"] == 'train':
                                self.env.Actions[tl_id]=actions[idx]
                            elif type(self.env.tested) == str: 
                                if 'strong' in self.env.tested :
                                    if self.env.Agents[tl_id].nb_stop_inb > self.env.Agents[tl_id].nb_mov_inb :
                                        self.env.Actions[tl_id] = 1
                                    else:
                                        self.env.Actions[tl_id] = 0
                                elif 'classic' in self.env.tested :
                                    self.env.Actions[tl_id] = None

                                elif 'random' in self.env.tested:
                                    self.env.Actions[tl_id]= random.choice([0,1])

                                elif self.env.baseline: 
                                    self.env.Actions[tl_id] = 1

                            elif self.env.baseline: 
                                self.env.Actions[tl_id] = 1

            else:
                forced_mask[idx] = 1
                if self.env.env_params.additional_params["policy"] != 'binary': 
                    self.env.Actions[tl_id] = self.env.Agents[tl_id].current_phase_idx   
                    forced_value[idx] = self.env.Agents[tl_id].current_phase_idx   
                else:
                    self.env.Actions[tl_id] = 0
                    forced_value[idx] = 0

            if self.env.env_params.additional_params['correct_actions'] and self.env.env_params.additional_params['Policy_Type'] == "Q_Learning":
                # IF USING "REAL ACTIONS ONLY" WE TRAIN USING THE ACTIONS VECTOR WITH ACTIONS THAT ARE ACTUALLY TAKEN                     
                actions[idx]=self.env.Actions[tl_id]

            
        # SAVE STEP 
        if self.env.env_params.additional_params['mode'] == 'train' :
            if self.env.step_counter > self.env.env_params.additional_params['wait_n_steps']:            
                if 'critic' in self.env.env_params.additional_params['Policy_Type'].lower() and 'actor' in self.env.env_params.additional_params['Policy_Type'].lower():
                    pass                     
                elif not self.env.baseline:
                    self.env.all_graphs.append([self.env.current_graphs[self.env.graph_of_interest] if self.env.env_params.additional_params['GCN'] else self.env.state,labels, actions, choices, forced_mask, forced_value, action_size]) # choices      

        return self.env.Actions        






    def print_graph_state(self):
        print(self.env.env_params.additional_params['graph_of_interest'], self.env.current_graphs[self.env.env_params.additional_params['graph_of_interest']],  "\n\n\n\n\n\n\n number of nodes : ", self.env.current_graphs[self.env.env_params.additional_params['graph_of_interest']].number_of_nodes(),  "\n\n\n\n\n\n\n NDATA", self.env.current_graphs[self.env.env_params.additional_params['graph_of_interest']].ndata,  "\n\n\n\n\n\n\n EDATA", self.env.current_graphs[self.env.env_params.additional_params['graph_of_interest']].edata)

        
        
        

    def setup_state_variables(self):

        # ARBITRARY NUMBER OF VEHICLES REPRESENTED ON A LANE NODE 
        if self.env.env_params.additional_params['veh_state'] and self.env.env_params.additional_params['lane_node_state']:
            for veh_idx in range(self.env.env_params.additional_params['num_observed']):
                for var_name, var_dim in self.env.env_params.additional_params['lane_per_veh_vars'].items():
                    self.env.env_params.additional_params['lane_vars'][str(str(veh_idx) + "_" + var_name)] = var_dim # SPEED

        # ARBITRARY NUMBER OF NEXT_PHASES REPRESENTED ON A CONNECTION NODE 
        if self.env.env_params.additional_params['phase_state'] :
            for phase_idx in range(self.env.env_params.additional_params['num_observed_next_phases']):
                for var_name, var_dim in self.env.env_params.additional_params['connection_per_phase_vars'].items():
                    self.env.env_params.additional_params['connection_vars'][str(str(phase_idx) + "_" + var_name)] = var_dim # SPEED


    def setup_classes_and_networks(self):

        #INITIALIZE OBJECTS 
        self.env.Nodes_connections = collections.OrderedDict()
        self.env.Agent = Agent
        self.env.Agent.Policy_Type = self.env.env_params.additional_params['Policy_Type']
        self.env.Lane = Lane
        self.env.Lane.veh_state = self.env.env_params.additional_params['veh_state']
        self.env.Lane.n_observed = self.env.env_params.additional_params['num_observed']
        self.env.Lane.reward_type = self.env.env_params.additional_params['reward_type']
        self.env.Lane.mode = self.env.env_params.additional_params['mode']
        self.env.Lane.save_extended_training_stats = self.env.env_params.additional_params['save_extended_training_stats']
        self.env.Lane.max_dist_queue = self.env.env_params.additional_params['max_dist_queue']
        self.env.Lane.min_dist_delay = self.env.env_params.additional_params['min_dist_delay']
        self.env.shortest_paths = {}
        self.env.full_lane_connections=[]
        self.env.Agents = collections.OrderedDict()
        self.env.center = {}
        self.env.lanes = []
        
        


        # CREATE AGENTS (TSCs) INSTANCES AND RELEVANT CONNECTIONS OBJECTS
        for lane_id in self.env.traci_connection.lane.getIDList():
            if ":" not in lane_id:
                self.env.lanes.append(lane_id)
                

        for tl_idx, tl_id in enumerate(self.env.traci_connection.trafficlight.getIDList()):
            self.env.Agents[tl_id]=Agent(tl_id)
            self.env.Agents[tl_id].phases_defs = self.env.traci_connection.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0].getPhases()
            
            
            
            
            self.env.Agents[tl_id].phases_defs = list(self.env.Agents[tl_id].phases_defs)
            

            for idx, phase in enumerate(self.env.Agents[tl_id].phases_defs):
                phase.idx = idx
            self.env.Agents[tl_id].orig_phases_defs = copy.deepcopy(self.env.Agents[tl_id].phases_defs)
            self.env.Agents[tl_id].n_phases = len(self.env.Agents[tl_id].phases_defs)
            self.env.Agents[tl_id].unordered_connections_trio = []
            self.env.Agents[tl_id].distance_dic = collections.OrderedDict()
            self.env.Nodes_connections[tl_id] = []
            
            if tl_idx == 0:
                self.env.max_phase_len = self.env.Agents[tl_id].n_phases
            elif self.env.Agents[tl_id].n_phases > self.env.max_phase_len:
                self.env.max_phase_len = self.env.Agents[tl_id].n_phases

                
                
            for link in self.env.traci_connection.trafficlight.getControlledLinks(tl_id):
                self.env.Agents[tl_id].unordered_connections_trio.append(link)
            for link in self.env.traci_connection.trafficlight.getControlledLinks(tl_id) :
                if link[0][0] not in [x[0] for x in self.env.Nodes_connections[tl_id]]:
                    self.env.Nodes_connections[tl_id].append((link[0][0], "inbound"))
                if link[0][1] not in [x[0] for x in self.env.Nodes_connections[tl_id]]:
                    self.env.Nodes_connections[tl_id].append((link[0][1], "outbound"))
                self.env.Agents[tl_id].inb_lanes.append(link[0][0])
                self.env.Agents[tl_id].outb_lanes.append(link[0][1])
                name = str(str(link[0][0])+"_"+str(link[0][1]))
                self.env.center[name]=link[0][2]
                self.env.Agents[tl_id].complete_controlled_lanes.append(link[0][0])
                self.env.Agents[tl_id].complete_controlled_lanes.append(link[0][1])
                self.env.Agents[tl_id].complete_controlled_lanes.append(link[0][2])
                self.env.full_lane_connections.append([link[0][0],link[0][2],1])
                self.env.full_lane_connections.append([link[0][2],link[0][1],1])
                self.env.Agents[tl_id].connections_trio.append(link)

            self.env.Agents[tl_id].inb_lanes = list(set(self.env.Agents[tl_id].inb_lanes))
            self.env.Agents[tl_id].outb_lanes = list(set(self.env.Agents[tl_id].outb_lanes))
            self.env.Agents[tl_id].complete_controlled_lanes = set(self.env.Agents[tl_id].complete_controlled_lanes)

            
            
        self.env.env_params.additional_params['n_tls'] = len(self.env.Agents)
        
        
        # GET SETS OF CENTRAL LANES (INTERIOR PART OF AN INTERSECTION) AND NORMAL LANES (LINKING INTERSECTIONS)
        self.env.central_lanes = set(self.env.center.values())
        self.env.lanes = set(self.env.lanes)

        # DICTIONNARY DEFINING IDENTIFIER OF EVERY TYPE OF NODE IN THE DIFFERENT NETWORKS
        self.env.nodes_types = {1: 'tl', 2: 'lane', 3: 'veh', 4: 'edge', 5: 'connection', 6: 'phase'}

        # FOR PARRALLEL COMPUTATION TO BE PERFORMABLE, NODE STATES HAVE TO BE OF THE SAME SIZE. WE THEREFORE DEFINE THE NODE_STATE_SIZE AS THE MAXIMUM STATE SIZE AMONG EVERY TYPE OF NODES
        self.env.node_state_size = max(sum(self.env.env_params.additional_params["tl_vars"].values()), sum(self.env.env_params.additional_params["lane_vars"].values()), sum(self.env.env_params.additional_params["edge_vars"].values()), sum(self.env.env_params.additional_params["connection_vars"].values()), sum(self.env.env_params.additional_params["phase_vars"].values()))

        if self.env.env_params.additional_params['veh_as_nodes']:
            self.env.node_state_size = max(self.env.node_state_size, sum(self.env.env_params.additional_params["veh_vars"].values()))


        # WE GENERATE ALL THE NETWORKS INCLUDED IN ('generated_graphs') FOR DEEP GRAPH LIBRARY 
        if self.env.seed  % self.env.n_workers == 0 :
            s = time.time()
        self.generate_networks()     

        # WE GET PREVIOUS AND FOLLOWING LANES FOR EVERY LANE (WILL BE USED TO IDENTIFY STARTING LANES AND ENDING LANES FOR VALID ROUTES)
        for idx,tl_id in enumerate(self.env.traci_connection.trafficlight.getIDList()):           
            
            
            for link in self.env.traci_connection.trafficlight.getControlledLinks(tl_id) :


                if tl_id not in [x[0] for x in self.env.Nodes_connections[link[0][0]]]:                    
                    self.env.Nodes_connections[link[0][0]].append((tl_id, "outbound"))

                if tl_id not in [x[0] for x in self.env.Nodes_connections[link[0][1]]]:     
                    self.env.Nodes_connections[link[0][1]].append((tl_id, "inbound"))


                self.env.Lanes[link[0][0]].next_tl = tl_id
                self.env.Lanes[link[0][1]].previous_tl = tl_id                

        if not self.env.env_params.additional_params['gen_trips_before_exp'] or (self.env.env_params.additional_params['mode'] == 'train' and self.env.env_params.additional_params['distance_gamma'] not in [None,0]):
            for idx, lane_id in enumerate(self.env.lanes):
                for idx2, lane_id2 in enumerate(self.env.lanes):
                    if lane_id2 not in self.env.Lanes[lane_id].distance_dic:
                        if nx.has_path(self.env.edge_graph, source=lane_id[:-2],target=lane_id2[:-2]):
                            shortest_paths = [p for p in nx.all_shortest_paths(self.env.edge_graph,source=lane_id[:-2],target=lane_id2[:-2])]
                            distance = len(shortest_paths[0]) -1
                        else:
                            distance = 1e3
                        self.env.Lanes[lane_id].distance_dic[lane_id2]= distance
                        self.env.Lanes[lane_id2].distance_dic[lane_id]= distance

                    is_valid_route = False  
                    if self.env.env_params.additional_params["grid"]:
                        if lane_id in self.env.lanes and lane_id2 in self.env.lanes and not self.env.Lanes[lane_id].inb_adj_lanes and not self.env.Lanes[lane_id2].outb_adj_lanes and set([i for i in lane_id]) == set([i for i in lane_id2]):
                            if ":" not in lane_id and ':' not in lane_id2:
                                is_valid_route = True



                    else:
                        if lane_id in self.env.lanes and lane_id2 in self.env.lanes and not self.env.Lanes[lane_id].inb_adj_lanes and not self.env.Lanes[lane_id2].outb_adj_lanes and lane_id.split('_')[0].split('-')[-1] != lane_id2.split('_')[0].split('-')[-1]:
                            if ":" not in lane_id and ':' not in lane_id2:
                                is_valid_route = True

                    if is_valid_route:
                        if nx.has_path(self.env.lane_graph, source=lane_id,target=lane_id2):
                            trip_name = str("route_" + lane_id[:-2] + "_" + lane_id2[:-2])     
                            if trip_name not in self.env.trip_names:  
                                self.env.trip_names.append(trip_name)
                                shortest_paths = [[edge[:-2] for edge in path] for path in nx.all_shortest_paths(self.env.lane_graph,source=lane_id,target=lane_id2)]
                                distance = len(shortest_paths[0]) -1

                                self.env.shortest_paths[trip_name] = shortest_paths

                                self.env.entering_edges.add(lane_id[:-2])
                                self.env.leaving_edges.add(lane_id2[:-2])                             


 
            # COMPUTE DISTANCES (FOR REWARD COMPUTATION) AND SHORTEST PATHS (TO CREATE SENSIBLE ROUTES FOR VEHICLES)
            for idx1, tl_id in enumerate(self.env.traci_connection.trafficlight.getIDList()):

                for idx2, lane_id in enumerate(self.env.lanes):

                    for idx3, controlled_lane in enumerate(self.env.Agents[tl_id].complete_controlled_lanes):

                        if lane_id in self.env.lanes and controlled_lane in self.env.lanes:
                                distance = self.env.Lanes[controlled_lane].distance_dic[lane_id]

                                if lane_id not in self.env.Agents[tl_id].distance_dic:
                                    self.env.Agents[tl_id].distance_dic[lane_id] = distance
                                elif distance < self.env.Agents[tl_id].distance_dic[lane_id]:
                                    self.env.Agents[tl_id].distance_dic[lane_id] = distance



                # WE CREATE A DISCOUNT VECTOR (BASED ON DISTANCES AND DISTANCE GAMMA) THAT WILL ENABLE EFFICIENT COMPUTATION OF THE REWARD FOR EVERY AGENT BASED ON NETWORK BASED DISTANCE 

                self.env.Agents[tl_id].distance_vector = np.asarray(list(collections.OrderedDict(sorted(self.env.Agents[tl_id].distance_dic.items(), key=lambda t: t[0])).values()))

                self.env.Agents[tl_id].discount_vector = self.env.env_params.additional_params["distance_gamma"]**self.env.Agents[tl_id].distance_vector

            for lane_id in self.env.Lanes:

                self.env.Nodes_connections[lane_id] = sorted(self.env.Nodes_connections[lane_id], key = lambda t: (t))
                self.env.Lanes[lane_id].distance_vector = np.asarray(list(collections.OrderedDict(sorted(self.env.Lanes[lane_id].distance_dic.items(), key=lambda t: t[0])).values()))
                self.env.Lanes[lane_id].discount_vector = self.env.env_params.additional_params["distance_gamma"]**self.env.Lanes[lane_id].distance_vector


        self.env.Nodes_connections = collections.OrderedDict(sorted(self.env.Nodes_connections.items(), key = lambda t: (-int(t[0].count('_')),t[0])))

        self.env.Lanes_vector_ordered = sorted(list(self.env.Lanes.keys()), key=lambda t: t)
        


    def generate_networks(self):
        self.env.lanes_in_graph = []

        # TL NETWORK (DIRECTED)   ( )
        if 'tl_graph' in self.env.env_params.additional_params['generated_graphs'] :
            self.env.tl_graph_dgl = dgl.DGLGraph()
            self.env.tl_graph_dgl.nodes_types = collections.OrderedDict()
            self.env.tl_graph_dgl.nodes_types['tl'] = 1 
            self.env.tl_graph_dgl.adresses_in_graph = collections.OrderedDict()
            self.env.tl_graph_dgl.norms = collections.OrderedDict()
            self.env.tl_graph_dgl.adresses_in_sumo = collections.OrderedDict()
            src = []
            dst = []
            tp = []
            norm = []
            node_type = []
            counter = 0
            for tl_id in self.env.Agents:    

                #CREATE NODE 
                if tl_id not in self.env.tl_graph_dgl.adresses_in_graph:
                    self.env.tl_graph_dgl.adresses_in_graph[tl_id] = counter
                    self.env.tl_graph_dgl.norms[tl_id] = [0]*2
                    self.env.tl_graph_dgl.adresses_in_sumo[str(counter)] = tl_id
                    node_type.append(1)
                    counter +=1 

                    #ADD SELF LOOP WITH TYPE AT THE END 
                    src.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id])
                    dst.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id])
                    self.env.tl_graph_dgl.norms[tl_id][-1] += 1     
                    tp.append(len(self.env.tl_graph_dgl.norms[tl_id])-1)                         



                for tl_id2 in self.env.Agents:
                    if tl_id2 not in self.env.tl_graph_dgl.adresses_in_graph:
                        self.env.tl_graph_dgl.adresses_in_graph[tl_id2] = counter
                        self.env.tl_graph_dgl.norms[tl_id2] = [0]*2
                        self.env.tl_graph_dgl.adresses_in_sumo[str(counter)] = tl_id2
                        node_type.append(1)
                        counter +=1

                        #ADD SELF LOOP WITH TYPE AT THE END 
                        src.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id2])
                        dst.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id2])
                        self.env.tl_graph_dgl.norms[tl_id2][-1] += 1     
                        tp.append(len(self.env.tl_graph_dgl.norms[tl_id2])-1)       

                    if list(set(self.env.Agents[tl_id].complete_controlled_lanes).intersection(self.env.Agents[tl_id2].complete_controlled_lanes)) and tl_id != tl_id2: # CHECK IF LINK EXISTS
                        src.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id])
                        dst.append(self.env.tl_graph_dgl.adresses_in_graph[tl_id2])
                        self.env.tl_graph_dgl.norms[tl_id2][0] += 1     
                        tp.append(0)

            for destination, t in zip(dst,tp):
                norm.append([(1/self.env.tl_graph_dgl.norms[self.env.tl_graph_dgl.adresses_in_sumo[str(destination)]][t])])


            num_nodes = counter

            self.env.tl_graph_dgl.add_nodes(num_nodes)
            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            edge_type = torch.LongTensor(tp)
            edge_norm = torch.FloatTensor(norm).squeeze()
            node_type = torch.LongTensor(node_type)

            self.env.tl_graph_dgl.add_edges(src,dst)                        
            self.env.tl_graph_dgl.edata.update({'rel_type': edge_type, 'norm': edge_norm})                        
            self.env.tl_graph_dgl.ndata.update({'node_type' : node_type})  
            
            

        self.env.lane_graph = nx.DiGraph()
        self.env.edge_graph = nx.Graph()

        if 'lane_graph' in self.env.env_params.additional_params['generated_graphs'] :            

            self.env.lane_graph_dgl = dgl.DGLGraph()
            self.env.lane_graph_dgl.nodes_types = collections.OrderedDict()
            self.env.lane_graph_dgl.nodes_types['lane'] = 1
            if self.env.env_params.additional_params['veh_as_nodes'] :
                self.env.lane_graph_dgl.nodes_types['veh'] = 2
            node_type = []
            self.env.lane_graph_dgl.adresses_in_graph = collections.OrderedDict()
            self.env.lane_graph_dgl.norms = collections.OrderedDict()
            self.env.lane_graph_dgl.adresses_in_sumo = collections.OrderedDict()

        self.env.full_graph = Graph() 

        for lane_connection in self.env.full_lane_connections:
            self.env.full_graph.add_edge(*lane_connection)    



        self.env.valid_lanes = (collections.OrderedDict(sorted(self.env.full_graph.edges.items(), key=lambda t: t[0]))).keys()
        self.env.valid_lanes = list(self.env.valid_lanes)

        self.env.Edges = collections.OrderedDict()
        self.env.Lanes = collections.OrderedDict()      
        self.env.lane_connections = []
        self.env.edge_connections = []

        for edge_id in self.env.traci_connection.edge.getIDList():
            self.env.Edges[edge_id]=Edge(edge_id)
            self.env.Edges[edge_id].next_edges = set()

            
            
        for lane_id in self.env.lanes:
                self.env.Lanes[lane_id]=Lane(lane_id)
                self.env.Lanes[lane_id].connection_count =0
                self.env.traci_connection.lane.setMaxSpeed(lane_id, self.env.env_params.additional_params['Max_Speed'])    
                self.env.Lanes[lane_id].max_speed = self.env.traci_connection.lane.getMaxSpeed(lane_id)
                self.env.Lanes[lane_id].length = self.env.traci_connection.lane.getLength(lane_id)
                
                if self.env.env_params.additional_params['grid']:
                    self.env.Lanes[lane_id].std_length = self.env.Lanes[lane_id].length/self.env.env_params.additional_params['grid_lane_length']
                else:
                    self.env.Lanes[lane_id].std_length = (self.env.traci_connection.lane.getLength(lane_id)-self.env.env_params.additional_params['min_lane_length'])/(self.env.env_params.additional_params['max_lane_length']-self.env.env_params.additional_params['min_lane_length'])
                self.env.Lanes[lane_id].distance_dic = collections.OrderedDict()

        counter = 0


        if 'lane_graph' in self.env.env_params.additional_params['generated_graphs'] :
            src = []
            dst = []
            tp = []
            norm = []
        for idx, lane_id in enumerate(self.env.lanes):
            
                self.env.lane_graph.add_node(lane_id)   
                self.env.edge_graph.add_node(lane_id[:-2])

                if 'lane_graph' in self.env.env_params.additional_params['generated_graphs'] :
                    if lane_id not in self.env.lane_graph_dgl.adresses_in_graph:
                        self.env.lanes_in_graph.append(lane_id)
                        self.env.lane_graph_dgl.adresses_in_graph[lane_id] = counter
                        self.env.lane_graph_dgl.norms[lane_id] = [0]*3
                        self.env.lane_graph_dgl.adresses_in_sumo[str(counter)] = lane_id
                        node_type.append(1)
                        counter += 1 

                        #ADD SELF LOOP WITH TYPE AT THE END 
                        src.append(self.env.lane_graph_dgl.adresses_in_graph[lane_id])
                        dst.append(self.env.lane_graph_dgl.adresses_in_graph[lane_id])
                        self.env.lane_graph_dgl.norms[lane_id][-1] += 1     
                        tp.append(len(self.env.lane_graph_dgl.norms[lane_id])-1)  

                self.env.Nodes_connections[lane_id] = []
                edge_id = self.env.traci_connection.lane.getEdgeID(lane_id)

                try:
                    self.env.Edges[edge_id]
                except:
                    self.env.Edges[edge_id] = Edges(edge_id)



                for idx,info_connection in enumerate(list(self.env.traci_connection.lane.getLinks(lane_id))):      

                    if 'lane_graph' in self.env.env_params.additional_params['generated_graphs'] :
                        if info_connection[0] not in self.env.lane_graph_dgl.adresses_in_graph:  
                            self.env.lane_graph_dgl.adresses_in_graph[info_connection[0]] = counter
                            self.env.lane_graph_dgl.norms[info_connection[0]] = [0]*3
                            self.env.lane_graph_dgl.adresses_in_sumo[str(counter)] = info_connection[0]
                            node_type.append(2)
                            counter += 1 

                            #ADD SELF LOOP WITH TYPE AT THE END 
                            src.append(self.env.lane_graph_dgl.adresses_in_graph[info_connection[0]])
                            dst.append(self.env.lane_graph_dgl.adresses_in_graph[info_connection[0]])
                            self.env.lane_graph_dgl.norms[info_connection[0]][-1] += 1     
                            tp.append(len(self.env.lane_graph_dgl.norms[info_connection[0]])-1)                         

                        # vehicle flow
                        src.append(self.env.lane_graph_dgl.adresses_in_graph[lane_id])
                        dst.append(self.env.lane_graph_dgl.adresses_in_graph[info_connection[0]])
                        self.env.lane_graph_dgl.norms[info_connection[0]][0] +=1
                        tp.append(0)
                        # reverse flow
                        src.append(self.env.lane_graph_dgl.adresses_in_graph[info_connection[0]])
                        dst.append(self.env.lane_graph_dgl.adresses_in_graph[lane_id])
                        self.env.lane_graph_dgl.norms[lane_id][1] +=1
                        tp.append(1)

                    self.env.lane_graph.add_node(info_connection[0])                        
                    self.env.lane_graph.add_edge(lane_id,info_connection[0])                     
                    self.env.edge_graph.add_node(info_connection[0][:-2])                        
                    self.env.edge_graph.add_edge(lane_id[:-2],info_connection[0][:-2])                        

                    self.env.edge_connections.append((lane_id[:-2],info_connection[0][:-2],1))                           
                    self.env.lane_connections.append((lane_id,info_connection[0],1))   
                    self.env.Lanes[lane_id].outb_adj_lanes.append(info_connection[0])
                    next_edge = self.env.traci_connection.lane.getEdgeID(info_connection[0])
                    self.env.Edges[edge_id].next_edges.add(next_edge)
                    self.env.Edges[next_edge].prev_edges.add(edge_id)
                    self.env.Lanes[info_connection[0]].inb_adj_lanes.append(lane_id)   




        if 'lane_graph' in self.env.env_params.additional_params['generated_graphs'] :
            for destination, t in zip(dst,tp):
                norm.append([(1/self.env.lane_graph_dgl.norms[self.env.lane_graph_dgl.adresses_in_sumo[str(destination)]][t])])



            num_nodes = counter

            self.env.lane_graph_dgl.add_nodes(num_nodes)
            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            edge_type = torch.LongTensor(tp)
            edge_norm = torch.FloatTensor(norm).squeeze()
            node_type = torch.LongTensor(node_type)


            self.env.lane_graph_dgl.add_edges(src,dst)                        
            self.env.lane_graph_dgl.edata.update({'rel_type': edge_type, 'norm': edge_norm})                                        
            self.env.lane_graph_dgl.ndata.update({'node_type' : node_type})                            



        self.env.lane_connections=list(set(tuple(i) for i in self.env.lane_connections))                    
        self.env.routes = []
        self.env.trip_names = []
        self.env.entering_edges = set()
        self.env.leaving_edges = set()
        self.env.entering_edges_probs = []
        self.env.leaving_edges_probs = []

        if 'tl_lane_graph' in self.env.env_params.additional_params['generated_graphs'] :

            self.env.tl_lane_graph_dgl = dgl.DGLGraph()
            self.env.tl_lane_graph_dgl.nodes_types = collections.OrderedDict()
            self.env.tl_lane_graph_dgl.nodes_types['tl'] = 1
            self.env.tl_lane_graph_dgl.nodes_types['lane'] = 2
            if self.env.env_params.additional_params['veh_as_nodes'] :
                self.env.tl_lane_graph_dgl.nodes_types['veh'] = 3
            self.env.tl_lane_graph_dgl.adresses_in_graph = collections.OrderedDict()
            self.env.tl_lane_graph_dgl.norms = collections.OrderedDict()
            self.env.tl_lane_graph_dgl.adresses_in_sumo = collections.OrderedDict()

            counter = 0
            src = []
            dst = []
            tp = []
            norm = []
            node_type = []
            for tl_id in self.env.Agents:
                if tl_id not in self.env.tl_lane_graph_dgl.adresses_in_graph: 
                    self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id] = counter
                    self.env.tl_lane_graph_dgl.norms[tl_id] = [0]*6
                    self.env.tl_lane_graph_dgl.adresses_in_sumo[str(counter)] = tl_id
                    node_type.append(1)
                    counter+=1

                    #ADD SELF LOOP WITH TYPE AT THE END 
                    src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                    dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                    self.env.tl_lane_graph_dgl.norms[tl_id][-1] += 1     
                    tp.append(len(self.env.tl_lane_graph_dgl.norms[tl_id])-1)                              


                for lane_id in self.env.Agents[tl_id].inb_lanes:
                    if lane_id not in self.env.tl_lane_graph_dgl.adresses_in_graph:
                        self.env.lanes_in_graph.append(lane_id)
                        self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id] = counter
                        self.env.tl_lane_graph_dgl.norms[lane_id] = [0]*6
                        self.env.tl_lane_graph_dgl.adresses_in_sumo[str(counter)] = lane_id
                        node_type.append(2)
                        counter += 1 

                        #ADD SELF LOOP WITH TYPE AT THE END 
                        src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                        dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                        self.env.tl_lane_graph_dgl.norms[lane_id][-2] += 1     
                        tp.append(len(self.env.tl_lane_graph_dgl.norms[lane_id])-2)                              


                    # vehicle flow
                    src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                    dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                    # reverse flow
                    src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                    dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                    self.env.tl_lane_graph_dgl.norms[lane_id][0] +=1
                    tp.append(0)
                    self.env.tl_lane_graph_dgl.norms[tl_id][1] +=1                        
                    tp.append(1)                    


                for lane_id in self.env.Agents[tl_id].outb_lanes:
                    if lane_id not in self.env.tl_lane_graph_dgl.adresses_in_graph:
                        self.env.lanes_in_graph.append(lane_id)
                        self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id] = counter
                        self.env.tl_lane_graph_dgl.norms[lane_id] = [0]*6
                        self.env.tl_lane_graph_dgl.adresses_in_sumo[str(counter)] = lane_id
                        node_type.append(2)
                        counter += 1 

                        #ADD SELF LOOP WITH TYPE AT THE END 
                        src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                        dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                        self.env.tl_lane_graph_dgl.norms[lane_id][-2] += 1     
                        tp.append(len(self.env.tl_lane_graph_dgl.norms[lane_id])-2)                              


                    # vehicle flow
                    src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                    dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                    # reverse flow
                    src.append(self.env.tl_lane_graph_dgl.adresses_in_graph[lane_id])
                    dst.append(self.env.tl_lane_graph_dgl.adresses_in_graph[tl_id])
                    self.env.tl_lane_graph_dgl.norms[lane_id][2] +=1
                    tp.append(2)
                    self.env.tl_lane_graph_dgl.norms[tl_id][3] +=1                        
                    tp.append(3)                               


            for destination, t in zip(dst,tp):
                norm.append([(1/self.env.tl_lane_graph_dgl.norms[self.env.tl_lane_graph_dgl.adresses_in_sumo[str(destination)]][t])])



            num_nodes = counter


            self.env.tl_lane_graph_dgl.add_nodes(num_nodes)
            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            edge_type = torch.LongTensor(tp)
            edge_norm = torch.FloatTensor(norm).squeeze()
            node_type = torch.LongTensor(node_type)

            self.env.tl_lane_graph_dgl.add_edges(src,dst)                        
            self.env.tl_lane_graph_dgl.edata.update({'rel_type': edge_type, 'norm': edge_norm})                                              
            self.env.tl_lane_graph_dgl.ndata.update({'node_type' : node_type})                            




        if 'tl_connection_lane_graph' in self.env.env_params.additional_params['generated_graphs']: 
            self.env.tl_connection_lane_graph_dgl = dgl.DGLGraph()
            self.env.tl_connection_lane_graph_dgl.nodes_types = collections.OrderedDict()
            self.env.tl_connection_lane_graph_dgl.nodes_types['tl'] = 0
            self.env.tl_connection_lane_graph_dgl.nodes_types['lane'] = 1
            self.env.tl_connection_lane_graph_dgl.nodes_types['connection'] = 2
            if self.env.env_params.additional_params['veh_as_nodes'] :
                self.env.tl_connection_lane_graph_dgl.nodes_types['veh'] = 3
            self.env.tl_connection_lane_graph_dgl.adresses_in_graph = collections.OrderedDict()
            self.env.tl_connection_lane_graph_dgl.norms = collections.OrderedDict()
            self.env.tl_connection_lane_graph_dgl.adresses_in_sumo = collections.OrderedDict()

            counter = 0
            src = []
            dst = []
            tp = []
            norm = []
            node_type = []        


            # TLs
            for tl_id in self.env.Agents:

                # CREATE NODES
                if tl_id not in self.env.tl_connection_lane_graph_dgl.adresses_in_graph: 
                    self.env.tl_connection_lane_graph_dgl.adresses_in_graph[tl_id] = counter                  
                    self.env.tl_connection_lane_graph_dgl.norms[tl_id] = [0]*9
                    self.env.tl_connection_lane_graph_dgl.adresses_in_sumo[str(counter)] = tl_id
                    node_type.append(self.env.tl_connection_lane_graph_dgl.nodes_types['tl'])
                    counter+=1


                    #ADD SELF LOOP WITH TYPE AT THE END 
                    src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[tl_id])
                    dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[tl_id])
                    self.env.tl_connection_lane_graph_dgl.norms[tl_id][-1] += 1     
                    tp.append(len(self.env.tl_connection_lane_graph_dgl.norms[tl_id])-1)                         


                #LINKS/CONNECTIONS
                for link_idx,link in enumerate(self.env.Agents[tl_id].unordered_connections_trio):
                    link_name = str(tl_id+"_link_"+str(link_idx))

                    if link_name not in self.env.tl_connection_lane_graph_dgl.adresses_in_graph: 
                        # CREATE NODES 
                        self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name] = counter
                        self.env.tl_connection_lane_graph_dgl.norms[link_name] = [0]*9
                        self.env.tl_connection_lane_graph_dgl.adresses_in_sumo[str(counter)] = link_name
                        node_type.append(self.env.tl_connection_lane_graph_dgl.nodes_types['connection'])
                        counter+=1


                        #ADD SELF LOOP WITH TYPE AT THE END 
                        src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])
                        dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])
                        self.env.tl_connection_lane_graph_dgl.norms[link_name][-2] += 1     
                        tp.append(len(self.env.tl_connection_lane_graph_dgl.norms[link_name])-2) 



                    # CREATE LINKS  
                    # vehicle flow
                    src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[tl_id])
                    dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])
                    # reverse flow
                    src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])
                    dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[tl_id])

                    self.env.tl_connection_lane_graph_dgl.norms[link_name][0] +=1
                    tp.append(0)
                    self.env.tl_connection_lane_graph_dgl.norms[tl_id][1] +=1                        
                    tp.append(1)   


                    for dir_idx in range(2):                        
                        lane_id = link[0][dir_idx]

                        if lane_id not in self.env.tl_connection_lane_graph_dgl.adresses_in_graph: 
                            # CREATE NODES 
                            self.env.lanes_in_graph.append(lane_id)
                            self.env.tl_connection_lane_graph_dgl.adresses_in_graph[lane_id] = counter
                            self.env.tl_connection_lane_graph_dgl.norms[lane_id] = [0]*9
                            self.env.tl_connection_lane_graph_dgl.adresses_in_sumo[str(counter)] = lane_id
                            node_type.append(self.env.tl_connection_lane_graph_dgl.nodes_types['lane'])
                            counter+=1


                            #ADD SELF LOOP WITH TYPE AT THE END 
                            src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[lane_id])
                            dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[lane_id])
                            self.env.tl_connection_lane_graph_dgl.norms[lane_id][-3] += 1     
                            tp.append(len(self.env.tl_connection_lane_graph_dgl.norms[lane_id])-3)  



                        # CREATE LINKS 
                        src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])
                        dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[lane_id])
                        # reverse flow
                        src.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[lane_id])
                        dst.append(self.env.tl_connection_lane_graph_dgl.adresses_in_graph[link_name])


                        if dir_idx == 0:

                            self.env.tl_connection_lane_graph_dgl.norms[lane_id][2] +=1
                            tp.append(2)
                            self.env.tl_connection_lane_graph_dgl.norms[link_name][3] +=1                        
                            tp.append(3)     


                        elif dir_idx == 1:

                            self.env.tl_connection_lane_graph_dgl.norms[lane_id][4] +=1
                            tp.append(4)
                            self.env.tl_connection_lane_graph_dgl.norms[link_name][5] +=1                        
                            tp.append(5)     



            for destination, t in zip(dst,tp):
                norm.append([(1/self.env.tl_connection_lane_graph_dgl.norms[self.env.tl_connection_lane_graph_dgl.adresses_in_sumo[str(destination)]][t])])



            num_nodes = counter


            self.env.tl_connection_lane_graph_dgl.add_nodes(num_nodes)
            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            edge_type = torch.LongTensor(tp)
            edge_norm = torch.FloatTensor(norm).squeeze()
            node_type = torch.LongTensor(node_type).squeeze()

            self.env.tl_connection_lane_graph_dgl.add_edges(src,dst)                        
            self.env.tl_connection_lane_graph_dgl.edata.update({'rel_type': edge_type, 'norm': edge_norm})                                        
            self.env.tl_connection_lane_graph_dgl.ndata.update({'node_type' : node_type})                            



















        if 'full_graph' in self.env.env_params.additional_params['generated_graphs'] :        


            #FULL GRAPH : IN CONSTRUCTION 


            self.env.full_graph_dgl = dgl.DGLGraph()
            self.env.full_graph_dgl.nodes_types = collections.OrderedDict()
            self.env.full_graph_dgl.nodes_types['tl'] = 0
            self.env.full_graph_dgl.nodes_types['lane'] = 1
            self.env.full_graph_dgl.nodes_types['edge'] = 2
            self.env.full_graph_dgl.nodes_types['connection'] = 3
            self.env.full_graph_dgl.nodes_types['phase'] = 4
            if self.env.env_params.additional_params['veh_as_nodes'] :
                self.env.full_graph_dgl.nodes_types['veh'] = 5
            self.env.full_graph_dgl.adresses_in_graph = collections.OrderedDict()
            self.env.full_graph_dgl.norms = collections.OrderedDict()
            self.env.full_graph_dgl.adresses_in_sumo = collections.OrderedDict()

            counter = 0
            src = []
            dst = []
            tp = []
            norm = []
            node_type = []        



            # TLs
            for tl_id in self.env.Agents:

                # CREATE NODES
                if tl_id not in self.env.full_graph_dgl.adresses_in_graph: 
                    self.env.full_graph_dgl.adresses_in_graph[tl_id] = counter                  
                    self.env.full_graph_dgl.norms[tl_id] = [0]*21
                    self.env.full_graph_dgl.adresses_in_sumo[str(counter)] = tl_id
                    node_type.append(self.env.full_graph_dgl.nodes_types['tl'])
                    counter+=1


                    #ADD SELF LOOP WITH TYPE AT THE END 
                    src.append(self.env.full_graph_dgl.adresses_in_graph[tl_id])
                    dst.append(self.env.full_graph_dgl.adresses_in_graph[tl_id])
                    self.env.full_graph_dgl.norms[tl_id][-1] += 1     
                    tp.append(len(self.env.full_graph_dgl.norms[tl_id])-1)                         



                #PHASES 
                for phase_idx, phase in enumerate(self.env.Agents[tl_id].orig_phases_defs):


                    phase_name = str(tl_id+"_phase_"+str(phase_idx))
                    if phase_name not in self.env.full_graph_dgl.adresses_in_graph: 
                        # CREATE NODES 
                        self.env.full_graph_dgl.adresses_in_graph[phase_name] = counter
                        self.env.full_graph_dgl.norms[phase_name] = [0]*21
                        self.env.full_graph_dgl.adresses_in_sumo[str(counter)] = phase_name
                        node_type.append(self.env.full_graph_dgl.nodes_types['phase'])
                        counter+=1


                        #ADD SELF LOOP WITH TYPE AT THE END 
                        src.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])
                        dst.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])
                        self.env.full_graph_dgl.norms[phase_name][-2] += 1     
                        tp.append(len(self.env.full_graph_dgl.norms[phase_name])-2)   





                    # CREATE LINKS       
                    # vehicle flow
                    src.append(self.env.full_graph_dgl.adresses_in_graph[tl_id])
                    dst.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])
                    # reverse flow
                    src.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])
                    dst.append(self.env.full_graph_dgl.adresses_in_graph[tl_id])

                    self.env.full_graph_dgl.norms[phase_name][0] +=1
                    tp.append(0)
                    self.env.full_graph_dgl.norms[tl_id][1] +=1                        
                    tp.append(1)                    



                    #LINKS/CONNECTIONS
                    for link_idx,link in enumerate(self.env.Agents[tl_id].unordered_connections_trio):
                        link_name = str(tl_id+"_link_"+str(link_idx))

                        if link_name not in self.env.full_graph_dgl.adresses_in_graph: 
                            # CREATE NODES 
                            self.env.full_graph_dgl.adresses_in_graph[link_name] = counter
                            self.env.full_graph_dgl.norms[link_name] = [0]*21
                            self.env.full_graph_dgl.adresses_in_sumo[str(counter)] = link_name
                            node_type.append(self.env.full_graph_dgl.nodes_types['connection'])
                            counter+=1


                            #ADD SELF LOOP WITH TYPE AT THE END 
                            src.append(self.env.full_graph_dgl.adresses_in_graph[link_name])
                            dst.append(self.env.full_graph_dgl.adresses_in_graph[link_name])
                            self.env.full_graph_dgl.norms[link_name][-3] += 1     
                            tp.append(len(self.env.full_graph_dgl.norms[link_name])-3)   


                        # CREATE LINKS  
                        # vehicle flow
                        src.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])
                        dst.append(self.env.full_graph_dgl.adresses_in_graph[link_name])
                        # reverse flow
                        src.append(self.env.full_graph_dgl.adresses_in_graph[link_name])
                        dst.append(self.env.full_graph_dgl.adresses_in_graph[phase_name])

                        if phase.state[link_idx] == 'G':
                            self.env.full_graph_dgl.norms[link_name][2] +=1
                            tp.append(2)
                            self.env.full_graph_dgl.norms[phase_name][3] +=1
                            tp.append(3)                 

                        elif phase.state[link_idx] == 'g':
                            self.env.full_graph_dgl.norms[link_name][4] +=1                    
                            tp.append(4)    
                            self.env.full_graph_dgl.norms[phase_name][5] +=1                    
                            tp.append(5)                                  

                        elif phase.state[link_idx] == 'r':
                            self.env.full_graph_dgl.norms[link_name][6] +=1                    
                            tp.append(6)
                            self.env.full_graph_dgl.norms[phase_name][7] +=1                    
                            tp.append(7)    

                        elif phase.state[link_idx] == 'y':
                            self.env.full_graph_dgl.norms[link_name][8] +=1                    
                            tp.append(8)
                            self.env.full_graph_dgl.norms[phase_name][9] +=1                    
                            tp.append(9)    



                        for dir_idx in range(2):                        
                            lane_id = link[0][dir_idx]

                            if lane_id[:-2] not in self.env.full_graph_dgl.adresses_in_graph: 
                                edge_name = lane_id[:-2]
                                self.env.lanes_in_graph.append(lane_id)
                                self.env.full_graph_dgl.adresses_in_graph[edge_name] = counter
                                self.env.full_graph_dgl.norms[edge_name] = [0]*21
                                self.env.full_graph_dgl.adresses_in_sumo[str(counter)] = edge_name
                                node_type.append(self.env.full_graph_dgl.nodes_types['edge'])
                                counter+=1


                                #ADD SELF LOOP WITH TYPE AT THE END 
                                src.append(self.env.full_graph_dgl.adresses_in_graph[edge_name])
                                dst.append(self.env.full_graph_dgl.adresses_in_graph[edge_name])
                                self.env.full_graph_dgl.norms[edge_name][-5] += 1     
                                tp.append(len(self.env.full_graph_dgl.norms[edge_name])-5)  




                            if lane_id not in self.env.full_graph_dgl.adresses_in_graph: 
                                # CREATE NODES 
                                self.env.lanes_in_graph.append(lane_id)
                                self.env.full_graph_dgl.adresses_in_graph[lane_id] = counter
                                self.env.full_graph_dgl.norms[lane_id] = [0]*21
                                self.env.full_graph_dgl.adresses_in_sumo[str(counter)] = lane_id
                                node_type.append(self.env.full_graph_dgl.nodes_types['lane'])
                                counter+=1


                                #ADD SELF LOOP WITH TYPE AT THE END 
                                src.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])
                                dst.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])
                                self.env.full_graph_dgl.norms[lane_id][-4] += 1     
                                tp.append(len(self.env.full_graph_dgl.norms[lane_id])-4)  


                                    # CREATE LINKS
                                src.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])
                                dst.append(self.env.full_graph_dgl.adresses_in_graph[edge_name])
                                # reverse flow
                                src.append(self.env.full_graph_dgl.adresses_in_graph[edge_name])
                                dst.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])

                                self.env.full_graph_dgl.norms[edge_name][14] +=1
                                tp.append(14)
                                self.env.full_graph_dgl.norms[lane_id][15] +=1                        
                                tp.append(15)     



                            # CREATE LINKS 
                            src.append(self.env.full_graph_dgl.adresses_in_graph[link_name])
                            dst.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])
                            # reverse flow
                            src.append(self.env.full_graph_dgl.adresses_in_graph[lane_id])
                            dst.append(self.env.full_graph_dgl.adresses_in_graph[link_name])


                            if dir_idx == 0:

                                self.env.full_graph_dgl.norms[lane_id][10] +=1
                                tp.append(10)
                                self.env.full_graph_dgl.norms[link_name][11] +=1                        
                                tp.append(11)     


                            elif dir_idx == 1:

                                self.env.full_graph_dgl.norms[lane_id][12] +=1
                                tp.append(12)
                                self.env.full_graph_dgl.norms[link_name][13] +=1                        
                                tp.append(13)     



            for destination, t in zip(dst,tp):
                norm.append([(1/self.env.full_graph_dgl.norms[self.env.full_graph_dgl.adresses_in_sumo[str(destination)]][t])])



            num_nodes = counter


            self.env.full_graph_dgl.add_nodes(num_nodes)
            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            edge_type = torch.LongTensor(tp)
            edge_norm = torch.FloatTensor(norm).squeeze()
            node_type = torch.LongTensor(node_type).squeeze()

            self.env.full_graph_dgl.add_edges(src,dst)                        
            self.env.full_graph_dgl.edata.update({'rel_type': edge_type, 'norm': edge_norm})                                            
            self.env.full_graph_dgl.ndata.update({'node_type' : node_type})                            





        self.env.original_graphs = {}


        if 'tl_graph' in self.env.env_params.additional_params["generated_graphs"]:
            self.env.original_graphs["tl_graph"] = self.env.tl_graph_dgl 

        if 'lane_graph' in self.env.env_params.additional_params["generated_graphs"]:
            self.env.original_graphs["lane_graph"] = self.env.lane_graph_dgl     

        if 'tl_lane_graph' in self.env.env_params.additional_params["generated_graphs"]:
            self.env.original_graphs["tl_lane_graph"] = self.env.tl_lane_graph_dgl  

        if 'tl_connection_lane_graph' in self.env.env_params.additional_params["generated_graphs"]:
            self.env.original_graphs["tl_connection_lane_graph"] = self.env.tl_connection_lane_graph_dgl  

        if 'full_graph' in self.env.env_params.additional_params["generated_graphs"]:
            self.env.original_graphs["full_graph"] = self.env.full_graph_dgl  


        for graph_name, graph in self.env.original_graphs.items():
            graph.nodes_lists = collections.OrderedDict()
            for node_type, identifier in graph.nodes_types.items():
                new_filt = partial(filt, identifier = [identifier])
                graph.nodes_lists[node_type] = graph.filter_nodes(new_filt)


        if self.env.env_params.additional_params["veh_as_nodes"]:
            for graph_name, graph in self.env.original_graphs.items():
                if "lane" in graph_name or "full" in graph_name:
                    for norm in graph.norms.values():
                        norm.extend([0,0,0])  

        for graph_name, graph in self.env.original_graphs.items():
            if graph_name != "lane_graph":
                x = graph.subgraph(list(graph.nodes_lists['tl'])).parent_nid
                graph.parent_nid_ = x






    def setup_paths(self):
        for trip_name, paths in self.env.shortest_paths.items():
            counter = 0
            for route in paths:
                self.env.traci_connection.route.add(str(trip_name + "_" + str(counter)), route)  
                counter+=1

        self.env.entering_edges_probs = np.ones(len(self.env.entering_edges))
        self.env.leaving_edges_probs = np.ones(len(self.env.leaving_edges))




    def clean_repos(self):
        try:
            shutil.rmtree(self.env.env_params.additional_params["rendering_path"]) 
        except Exception as e:
            pass
        try:
            os.makedirs(self.env.env_params.additional_params["rendering_path"])           
        except Exception as e:
            pass

        
        
        
    def update_metrics(self):
        arrived_ids = self.env.traci_connection.simulation.getArrivedIDList()
        for veh_id in arrived_ids:
            if self.env.env_params.additional_params['mode'] == 'test':
                if veh_id in self.env.trips_dict:
                    self.env.trips_dict[veh_id]['finish_time'] =  self.env.traci_simulation_time -1
                    self.env.trips_dict[veh_id]['trip_duration'] = (self.env.trips_dict[veh_id]['finish_time'] - int(float(self.env.trips_dict[veh_id]['depart'])))
                    self.env.trips_dict[veh_id]['truly_finished'] = True
                    self.env.nb_completed_trips +=1
                    self.env.trips_to_complete.remove(veh_id)

            
        for lane_id in self.env.busy_lanes:
            if self.env.env_params.additional_params["mode"] == 'train':
                if self.env.env_params.additional_params["reward_type"] == 'delay':       
                    self.env.delay_vector[self.env.Lanes_vector_ordered.index(lane_id)] = self.env.Lanes[lane_id].delay                
                elif self.env.env_params.additional_params["reward_type"] == 'squared_delay':
                    self.env.squared_delay_vector[self.env.Lanes_vector_ordered.index(lane_id)] = self.env.Lanes[lane_id].squared_delay         
                elif self.env.env_params.additional_params["reward_type"] == 'queue_length':
                    self.env.queue_vector[self.env.Lanes_vector_ordered.index(lane_id)] =  self.env.Lanes[lane_id].nb_stop_veh
            self.env.total_delay_vector[self.env.Lanes_vector_ordered.index(lane_id)] = self.env.Lanes[lane_id].total_delay
            self.env.total_queue_vector[self.env.Lanes_vector_ordered.index(lane_id)] =  self.env.Lanes[lane_id].total_queue
                

        if self.env.env_params.additional_params["mode"] == 'train':
            if self.env.env_params.additional_params["reward_type"] == 'delay' :
                self.env.reward_vector = -np.asarray(self.env.delay_vector)
            elif self.env.env_params.additional_params["reward_type"] == 'squared_delay':
                self.env.reward_vector = -np.asarray(self.env.squared_delay_vector)
            elif self.env.env_params.additional_params["reward_type"] == 'queue_length':
                self.env.reward_vector = -np.asarray(self.env.queue_vector)   
                
            if self.env.step_counter > self.env.env_params.additional_params["wait_n_steps"]:                    

                self.env.global_reward += self.env.reward_vector.sum()

        if self.env.env_params.additional_params['mode'] == 'test' or self.env.env_params.additional_params['save_extended_training_stats']:
            if self.env.step_counter > self.env.env_params.additional_params["wait_n_steps"]:  
                self.env.global_delay -= np.asarray(self.env.total_delay_vector).sum()
                self.env.steps_delays.append(float(np.asarray(self.env.total_delay_vector).sum()))
                self.env.global_co2 -= self.env.co2
                self.env.steps_co2.append(float(self.env.co2))
                self.env.global_queues -= np.asarray(self.env.total_queue_vector).sum()
                self.env.steps_queues.append(float(np.asarray(self.env.total_queue_vector).sum()))
          
        
    def send_results(self):
        if self.env.env_params.additional_params['mode'] == 'test':
            self.env.tested_end.send((self.env.global_delay, self.env.global_queues, self.env.global_co2, self.env.nb_completed_trips))
            self.env.global_delay = 0 
            self.env.global_queues = 0
            self.env.global_co2 = 0
            self.env.nb_completed_trips = 0 # EASY 
        else:
            if self.env.greedy:
                self.env.greedy_reward_queue.put(float(self.env.global_reward))
            elif self.env.env_params.additional_params['save_extended_training_stats']:
                self.env.reward_queue.send((float(self.env.global_reward),float(self.env.global_delay),float(self.env.global_queues), float(self.env.global_co2), float(self.env.nb_completed_trips)))
                self.env.global_delay = 0 
                self.env.global_queues = 0
                self.env.global_co2 = 0
                self.env.nb_completed_trips = 0 # EASY 
            else:
                self.env.reward_queue.send(float(self.env.global_reward))           
                
            self.env.global_reward = 0
            
        
    def set_rendering_path(self):
        suffix = '/' + str(self.env.tested) + '/'
        self.env.rendering_path = str("sumo_rendering" + "/" + self.env.env_params.additional_params['tb_foldername'] + "/" + self.env.env_params.additional_params['tb_filename'] + suffix)
         
            
    def init_variables(self):
        self.env.baseline = False
        self.env.greedy = False

        if self.env.env_params.additional_params['mode'] == 'test' and self.env.tested != None:
            if type(self.env.tested) == str:
                if 'greedy' in self.env.tested.lower():
                    self.env.greedy = True
                else:
                    self.env.baseline = True
            else:
                self.env.baseline = True

            if type(self.env.tested) == int:
                self.env.env_params.additional_params["min_time_between_actions"] = self.env.tested

        self.env.graph_of_interest = self.env.env_params.additional_params["graph_of_interest"]
        graph_of_interest = self.env.graph_of_interest
        self.actions_counts = collections.OrderedDict()           
        for i in range(self.env.env_params.additional_params['n_actions']):
            self.actions_counts[i] = 0

        self.env.step_counter = 0
        self.env.all_graphs = []
        self.env.last_lane = {}
        
        
    def init_sumo_subscriptions_and_class_variables(self):
        
        self.env.traci_connection.simulation.subscribe(varIDs=[traci.constants.VAR_DEPARTED_VEHICLES_IDS])
        for idx,tl_id in enumerate(self.env.Agents):
            if type(self.env.tested) == str: 
                if 'strong' in self.env.tested :
                    self.env.Agents[tl_id].phases_scores = collections.OrderedDict()
                    for link in self.env.Agents[tl_id].unordered_connections_trio:
                        self.env.Lanes[link[0][0]].connection_count +=1

            self.env.traci_connection.trafficlight.subscribe(tl_id, [traci.constants.TL_CURRENT_PHASE, traci.constants.TL_RED_YELLOW_GREEN_STATE]) 
            self.env.Agents[tl_id].current_phase_idx = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[traci.constants.TL_CURRENT_PHASE]
            self.env.Agents[tl_id].current_phase = self.env.traci_connection.trafficlight.getSubscriptionResults(tl_id)[traci.constants.TL_RED_YELLOW_GREEN_STATE]    
            self.env.Agents[tl_id].max_idx =  self.env.Agents[tl_id].n_phases
            self.env.Agents[tl_id].cycle_duration = self.env.Agents[tl_id].max_idx -1               

            self.env.Agents[tl_id].time_since_last_action = 0
            self.env.Agents[tl_id].reset_rewards()

        for lane_id in self.env.Lanes:

            self.env.traci_connection.lane.subscribe(lane_id, [traci.constants.LAST_STEP_VEHICLE_ID_LIST, traci.constants.LANE_LINKS]) 
            self.env.Lanes[lane_id].reset_rewards()   

    def reset_metrics(self):
        if self.env.env_params.additional_params['mode'] == 'train':            
            self.env.reward_vector = [0] * len(self.env.Lanes)
            if self.env.env_params.additional_params['reward_type'] == 'delay':
                self.env.delay_vector = [0] * len(self.env.Lanes)    
            elif self.env.env_params.additional_params['reward_type'] == 'queue_length':
                self.env.queue_vector = [0] * len(self.env.Lanes)      
            elif self.env.env_params.additional_params['reward_type'] == 'squared_delay':
                self.env.squared_delay_vector = [0] * len(self.env.Lanes)                
         
        if self.env.env_params.additional_params['mode'] == 'test' or self.env.env_params.additional_params['save_extended_training_stats']:
            self.env.co2 = 0
            self.env.total_delay_vector = [0] * len(self.env.Lanes)    
            self.env.total_queue_vector = [0] * len(self.env.Lanes)  
                