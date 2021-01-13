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

# Make our DeepQ network
model = DeepQ(env)

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
         
        return False

# Train
steps, rewards = train(model, env, MSELoss(), Adam, render = args.render,
        episodes = 1000,
        experience_replay = True, experience_memory_size=1_000_000, batch_size=64,
        target_network = True, sync_every_steps = 1e4,
        gamma = 0.99, epsilon = 1.0, learning_rate = 5e-4,
        save_every=1000, on_episode_complete=solved)


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