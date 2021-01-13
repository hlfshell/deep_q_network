import numpy as np
import torch
from torch import nn
import gym

class DeepQ(nn.Module):

    def __init__(self, enviornment : gym.Env):
        super(DeepQ, self).__init__()

        # Our simple NN model checks the observation space
        # being passed, which will be our turn by turn state.
        # Our output is the action space.
        input_size = len(enviornment.reset())
        output_size = enviornment.action_space.n

        # Our input and output layers are defined by our 
        # input and output size, which is defined by our
        # environment. Otherwise a simple neural network
        # is fine here.
        self.model = nn.Sequential(
            # Input 
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            # Output
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.model(x)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
    
