import collections
import numpy as np

class Agent():
    connections_state = []

    def __init__(self, tl_id):
        self.reward = 0
        self.state = 0
        self.connection_rewards = collections.OrderedDict()
        self.connection_values = collections.OrderedDict()
        self.agent_id = tl_id
        self.inb_lanes = []
        self.outb_lanes = []
        self.connections_trio = []
        self.connections_info = []
        self.complete_controlled_lanes = []
        self.is_time_to_choose = False    
        
        
    def reset(self):
        return 0
    
    def reset_rewards(self):
        self.reward = 0 
        self.connection_rewards = collections.OrderedDict()
        for connection in self.connections_trio:
            self.connection_rewards[connection[0][0]] = 0 
            
    def get_reward(self, reward_vector):
        self.reward =  np.matmul(self.discount_vector,reward_vector)
        return self.reward
