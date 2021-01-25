import numpy as np
import torch
from torch import nn
import gym
from math import floor

class DeepQ_CNN(nn.Module):

    def __init__(self, enviornment : gym.Env, grayscale=False, downsize=False):
        super(DeepQ_CNN, self).__init__()

        self.grayscale = grayscale
        self.downsize = downsize

        # Our simple NN model checks the observation space
        # being passed, which will be our turn by turn state.
        # Our output is the action space.
        height, width, channels = enviornment.reset().shape
        output_size = enviornment.action_space.n

        # If we downsize by 2, modify the height and width
        if downsize:
            height /= 2
            width /= 2
        # If we go grayscale, reduce the channels to 1
        if grayscale:
            channels = 1

        # The Atari environments return images that are HxWxC
        # while Pytorch conv2d is expecting the shape of CxHxW.
        # This means I'll have to permute the tensor prior to
        # passing it. Note the difference moving forward.

        # We'll need to calculate the output size of our conv layers as
        # we will need the final size for our flattened layer
        # Our first input is the dimensions of our image (sans channels)
        conv_output_shape = (height, width)
        for i in range(2):
            conv_output_shape = calculate_conv2d_output_shape(conv_output_shape[0], conv_output_shape[1], kernel_size=(5,5))

        self.model = nn.Sequential(
            # Input
            nn.Conv2d(channels, 32, 5),

            nn.Conv2d(32, 32, 5),

            # nn.Conv2d(64, 128, 5),


            # Flatten our CNN output
            nn.Flatten(),

            # Now we can do a dense network
            nn.Linear(conv_output_shape[0]*conv_output_shape[1]*32, 256),
            nn.LeakyReLU(),

            # nn.Linear(256, 64),
            # nn.LeakyReLU(),

            # Output
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        return self.model(x)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

# The calculations within are pulled from the docs for pytorch conv2d, here:
# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
# Defaults are pulled from defaults of the pytorch function
def calculate_conv2d_output_shape(height, width, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1)):
    h = floor( ( (height + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) -1)/stride[0] ) + 1 )
    w = floor( ( (width + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) -1)/stride[1] ) + 1 )

    return (h, w)