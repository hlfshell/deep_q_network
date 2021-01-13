import argparse
import gym
from torch.optim import Adam
from torch.nn import MSELoss
from matplotlib import pylab as plt
import numpy as np
import torch

from model import DeepQ

# Parse incoming arguments for the trainer
parser = argparse.ArgumentParser()
parser.add_argument("environment", help="which environment to run training on")
parser.add_argument("model", help="Model file to run on environment")
parser.add_argument("--render", action="store_true", help="render the environment")
args = parser.parse_args()

# Instantiate our gym
env = gym.make(args.environment)

# Make our DeepQ network
model = DeepQ(env)
model.load(args.model)

rewards = []
steps = []

for episode in range(0, 100):
    done = False
    step = 0

    total_reward = 0

    state = env.reset()

    while not done:
        step += 1

        if args.render:
            env.render()

        # Convert our state to pytorch - ensure it's float
        state = torch.from_numpy(state).float()

        Q = model.forward(state)
        q_values = Q.data.numpy()
        action = np.argmax(q_values)

        state, reward, done, _ = env.step(action)
        total_reward += reward

        print(f"Episode {episode+1} | Step {step} | Reward: {total_reward}", end="\r")

    rewards.append(total_reward)
    steps.append(step)

print()

print(f"Average # of steps per episode: {sum(steps)/len(steps)} | Longest episode was {max(steps)} | Shortest episode was {min(steps)}")
print(f"Average reward per episode: {sum(rewards)/len(rewards)} | Highest reward was {max(rewards)} | Lowest reward was {min(rewards)}")
