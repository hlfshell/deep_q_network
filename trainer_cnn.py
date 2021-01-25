import argparse
import gym
from torch import Tensor
from torch.optim import Adam
from torch.nn import MSELoss, SmoothL1Loss
from matplotlib import pylab as plt
import numpy as np

from train import train
from model_cnn import DeepQ_CNN

# Parse incoming arguments for the trainer
parser = argparse.ArgumentParser()
parser.add_argument("environment", help="which environment to run training on")
parser.add_argument("--render", action="store_true", help="render the environment")
args = parser.parse_args()

# Instantiate our gym
env = gym.make(args.environment)

# Make our DeepQ network
model = DeepQ_CNN(env, downsize=True, grayscale=True)

# Solved is being passed to on_episode_complete, which is called
# upon each episode completion. From there, we will check to see
# if our environment has reached the appropriate "solved" condition
# if one exists
def solved(episode, step, steps, total_reward, rewards):
        solved = False

        # LunarLander solved if we are 100 consecutive episodes
        # with an average reward above 200
        if args.environment == "LunarLander-v2":
                if len(rewards) > 100 and sum(rewards[-100:])/100 >= 200:
                        solved = True
        # MountainCar-v0 is considered solved if you have 100
        # conesecutive episodes with an average reward over -110
        elif args.environment == "MountainCar-v0":
                if len(rewards) > 100 and sum(rewards[-100:])/100 >= -110:
                        solved = True

        # CartPole-v0 is considered solved if you have 100
        # consecutive episodes with an average reward over 195.
        elif args.environment == "CartPole-v0":
                if len(rewards) > 100 and sum(rewards[-100:])/100 >= 195:
                        solved = True

        if solved:
                print()
                print(f"Solved after {step} steps")
                return True

        print(f"Episode {episode+1} took {step} steps for a reward of {total_reward:.2f}. - REWARDS - Last 100: {sum(rewards[-100:])/len(rewards[-100:]) if len(rewards) > 0 else 0:.2f} - Last 10: {sum(rewards[-10:])/len(rewards[-10:]) if len(rewards) > 0 else 0:.2f}", end="\r")
         
        return False

def state_transform(state):
        # Transform image to black and white
        state = np.mean(state, axis=3)
        # Down sample - half the size!
        state = state[::, ::2, ::2]
        state = Tensor(state)
        # Here we are adding the channel dimension as pytorch expects it in the 1th dimension,
        # IE: (batch, channels, height, width)
        state = state.unsqueeze(dim=1)
        return state

# Train
steps, rewards = train(model, env, MSELoss(), Adam, render = args.render,
        episodes = 1000,
        experience_replay = True, experience_memory_size=1_000_000, batch_size=64,
        target_network = True, sync_every_steps = 1e4,
        gamma = 0.99, epsilon = 1.0, epsilon_minimum=0.10,
        epsilon_minimum_at_episode=500,
        learning_rate = 5e-4,
        on_episode_complete=solved, state_transform=state_transform)

print()
print("Training complete")

plt.figure(figsize=(10,7))
plt.plot(steps)
plt.xlabel("Episodes", fontsize=22)
plt.ylabel("Steps", fontsize=22)
plt.savefig("steps.png")

plt.figure(figsize=(10,7))
plt.plot(rewards)
plt.xlabel("Episodes", fontsize=22)
plt.ylabel("Rewards", fontsize=22)
plt.savefig("rewards.png")

filepath = "deep.q.model.pt"

model.save(filepath)
print(f"Model saved to {filepath}")