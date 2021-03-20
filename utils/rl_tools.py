import torch
import numpy as np
import torch.nn.functional as F

def gaussian_probability(sigma, mu, data):
    data = data.unsqueeze(1).unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)
    
def sample(pi, sigma, mu):
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample
    

def mdn_loss(pi, sigma, mu, target):
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = - torch.logsumexp(prob, dim = 1, keepdim=False, out=None)
    return torch.mean(nll)

class Memory():
    def __init__(self, manager, max_size):
        self.manager = manager
        self.buffer = self.manager.list()
        self.max_size = max_size
        
    def add(self, experience):
        if len(self.buffer)>=self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

