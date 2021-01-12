import argparse
import gym
from torch.optim import Adam
from torch.nn import MSELoss, SmoothL1Loss
from matplotlib import pylab as plt

from train import train
from model import DeepQ

# Parse incoming arguments for the trainer
parser = argparse.ArgumentParser()
parser.add_argument("environment", help="which environment to run training on")
parser.add_argument("--render", action="store_true", help="render the environment")
args = parser.parse_args()

# Instantiate our gym
env = gym.make(args.environment)
env.seed(0)

# Make our DeepQ network
model = DeepQ(env)

# Make our trainer
# Train!
steps, rewards = train(model, env, MSELoss(), Adam, render = args.render,
        episodes = 1000,
        experience_replay = True, experience_memory_size=1_000_000, batch_size=64,
        target_network = False, sync_every_steps = 50,
        gamma = 0.99, epsilon = 1.0, learning_rate = 0.001) #5e-4)


print("Training complete")

plt.figure(figsize=(10,7))
plt.plot(steps)
plt.xlabel("Games", fontsize=22)
plt.ylabel("Steps", fontsize=22)
plt.savefig("steps.png")

plt.figure(figsize=(10,7))
plt.plot(rewards)
plt.xlabel("Games", fontsize=22)
plt.ylabel("Reward", fontsize=22)
plt.savefig("rewards.png")

filepath = "deep.q.model.pt"

model.save(filepath)
print(f"Model saved to {filepath}")