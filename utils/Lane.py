import numpy as np 
import collections


class Lane():
    
    def __init__(self, lane_id):

        self.lane_id= lane_id
        self.reward = 0
        self.inb_adj_lanes = []
        self.outb_adj_lanes = []


    def reset_rewards(self):
        self.reward = 0
        
        
        
    def get_reward(self):
        if Lane.reward_type == 'queue_length':
            return -self.nb_stop_veh
        elif Lane.reward_type == 'delay':
            return -self.delay
        elif Lane.reward_type == 'squared_delay':
            return -self.squared_delay

    def reset(self):
        self.position_closest_vehicle = self.length
        self.target = 0
        self.total_queue = 0
        self.total_delay = 0
        if Lane.veh_state:
            self.state = [0] * ((2 * Lane.n_observed) + 2)
        else:
            self.state = [0] * 2
            
            
        self.nb_mov_veh = 0
        self.nb_stop_veh = 0
        if Lane.mode == 'test':
            self.delay = 0
        
        elif Lane.mode == 'train':
            if Lane.reward_type == 'delay':
                self.delay = 0
            elif Lane.reward_type == 'queue_length':
                self.nb_stop_veh = 0
                self.nb_mov_veh = 0
            elif Lane.reward_type == 'squared_delay':
                self.squared_delay = 0


             
    def update_lane_state_and_reward(self,veh_speed, veh_max_speed, position, veh_state = False, ignored = False):

        veh_queue = 0
        delay = 0
        if Lane.mode == 'test' or Lane.save_extended_training_stats:
            if veh_speed < 0.1:
                veh_queue = 1
                self.total_queue +=1
                if Lane.mode == 'test' and (self.length - position) <= Lane.max_dist_queue and position <= self.length:   # and position >= Lane.min_dist_delay:
                    self.nb_stop_veh +=1 
            else:
                if Lane.mode == 'test' and (self.length - position) <= Lane.max_dist_queue and position <= self.length:   # and position >= Lane.min_dist_delay:
                    self.nb_mov_veh +=1
                    
            reachable_speed = min(veh_max_speed,self.max_speed)
            delay = ( (reachable_speed - veh_speed) / reachable_speed)
            self.total_delay+= delay             

            
        if position <= self.length:
            if Lane.mode == 'train': 
                if Lane.reward_type == 'queue_length':
                    if (self.length - position) <= Lane.max_dist_queue: #and position >= Lane.min_dist_delay:
                        if veh_speed < 0.1:
                            self.nb_stop_veh +=1
                        elif position <= self.length:
                            self.nb_mov_veh +=1
                elif Lane.reward_type == 'delay':
                    if position >= Lane.min_dist_delay:
                        if not Lane.save_extended_training_stats:
                            reachable_speed = min(veh_max_speed,self.max_speed)
                            delay = ( (reachable_speed - veh_speed) / reachable_speed)
                        self.delay+= delay 
                elif Lane.reward_type == 'squared_delay':
                    if position >= Lane.min_dist_delay:
                        if not Lane.save_extended_training_stats:
                            reachable_speed = min(veh_max_speed,self.max_speed)
                            delay = ( (reachable_speed - veh_speed) / reachable_speed)
                        self.squared_delay += delay**2



            if (self.length - position) <= Lane.max_dist_queue:
                if self.state[0] == 0:
                    self.state[1] = veh_speed
                self.state[0] +=1
                self.state[1] = self.state[1] + (veh_speed - self.state[1])/self.state[0]


                position = (self.length - position) / self.length 
                if position < self.position_closest_vehicle:
                    self.position_closest_vehicle = position
                    self.target = veh_speed


                if Lane.veh_state : 
                    for idx,i in enumerate(self.state[3::2]):
                        if position <= i:
                            self.state.insert(2*(idx+1),position)
                            self.state.insert(2*(idx+1),veh_speed)
                            if len(self.state) > ((2 * Lane.n_observed) + 2):
                                self.state.pop()
                                self.state.pop()
                            break

                        
                        
        return delay, veh_queue
