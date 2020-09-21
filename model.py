import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(net, low=-3e-2, high=3e-2):
    for param in net.parameters():
        param.data.uniform_(low, high)


class Actor(nn.Module):
    """
    Actor Network
    """
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        initialize_weights(self)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x).tanh()


class Critic(nn.Module):
    """
    Critic Network
    """
    def __init__(self, state_size, action_size, num_atoms, seed, fcs1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, num_atoms)
        initialize_weights(self)

    def forward(self, state, action, log=False):
        xs = F.relu(self.fcs1(state))
        x = torch.cat([xs, action], dim=1)
        x = F.relu(self.fc2(x))
        if log:
            return F.log_softmax(self.fc3(x), dim=-1)
        else:
            return F.softmax(self.fc3(x), dim=-1)
