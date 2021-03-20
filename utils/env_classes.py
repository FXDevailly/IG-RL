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
from IPython.display import clear_output
import glob
import atexit
import time
import traceback


class Params():
    def __init__(self):
        pass
    
class Env():
    
    def __init__(self, params, connection):
        self.time_counter = 0
        self.traci_connection = connection
        self.env_params = Params()
        self.env_params.additional_params = params

    def step(self, rl_actions):
        if self.env_params.additional_params["clear"]:              
            clear_output()         
            
        for _step_ in range(self.env_params.additional_params['sims_per_steps']):
            self.time_counter += 1

            
            if rl_actions and _step_ == 0:
                
                self.apply_rl_actions(rl_actions)


            # advance the simulation in the simulator by one step
            self.traci_connection.simulationStep()  
            self.save_render()


    def apply_rl_actions(self, rl_actions):
        if rl_actions is not None:
            for idx, (tl_id, action) in enumerate(rl_actions.items()):
                action = rl_actions[tl_id]
                if action is not None:                
                    if self.env_params.additional_params["policy"] != 'binary':
                        self.traci_connection.trafficlight.setPhase(tl_id,action)                        
                    else:
                        action = action > 0.0
                        if action:
                            if self.Agents[tl_id].current_phase_idx == self.Agents[tl_id].max_idx -1 :
                                self.traci_connection.trafficlight.setPhase(tl_id,0) 
                            else:
                                self.traci_connection.trafficlight.setPhase(tl_id,self.Agents[tl_id].current_phase_idx+1)




    def save_render(self):   
        # EVERY "FREQ", SAVE "LENGTH" STEPS
        if self.env_params.additional_params["save_render"] and self.step_counter >= self.env_params.additional_params["wait_n_steps"]:
            if self.env_params.additional_params["viz_exp_length"] > (self.step_counter - self.env_params.additional_params["wait_n_steps"]) % self.env_params.additional_params["viz_exp_frequency"] >  0 :
                self.traci_connection.gui.screenshot("View #0", self.rendering_path +str(self.capture_counter)+".png")
                self.capture_counter +=1
            elif (self.step_counter - self.env_params.additional_params["wait_n_steps"]) % self.env_params.additional_params["viz_exp_frequency"] ==  self.env_params.additional_params["viz_exp_length"] :

                self.capture_counter = 0
                os.system(str("ffmpeg -framerate 3 -i " + str(self.rendering_path +  "%d.png ") + str(self.rendering_path + "video_" + str(self.step_counter) + ".webm")))
                filelist=glob.glob(str(self.rendering_path) + "*.png")
                for file in filelist:
                    os.remove(file)
        

    def initialize_additional_params(self):
        self.graph_of_interest = self.env_params.additional_params['graph_of_interest']
        self.steps_done = 0 
        self.env_params.additional_params['tb_filename'] = str(self.env_params.additional_params['mode'] + "_" + self.env_params.additional_params['tb_filename'])
        if self.env_params.additional_params["random_objectives"]:
            self.objectives = ['column', 'line', 'full']
            for tl_id in self.Agents:  
                objective = self.r.choice(self.objectives)
                self.Agents[tl_id].objective = objective 

def create_DNN_model():
    env.model = DNN_IQL(double = env.env_params.additional_params['double_DQN'],
                        target_model = env.env_params.additional_params['target_model_update_frequency'],
                        n_actions = env.env_params.additional_params["n_actions"],
                        dueling = env.env_params.additional_params['dueling_DQN'], 
                        tl_input_dims = get_input_dims(), 
                        activation = F.elu, 
                        n_workers = env.n_workers,
                        batch_size = env.env_params.additional_params["batch_size"])
    env.model.train()