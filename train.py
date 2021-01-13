import torch
from torch import optim
from random import random, sample
import numpy as np
from collections import deque
import copy

def train(model, environment, loss_function, optimizer_function,
            learning_rate=1e-4, episodes=5000, epsilon=1, gamma=0.95,
            experience_replay=True, experience_memory_size=1000,
            target_network=True, batch_size=50, sync_every_steps=500,
            save_every=None,
            on_episode_complete=None,
            render=False):

    rewards = []
    steps = []

    # If needed, prep our experience replay buffer
    if experience_replay:
        experience_memory = deque(maxlen=experience_memory_size)

    # If needed, prep our target network
    if target_network:
        target_model = copy.deepcopy(model)
        target_model.load_state_dict(model.state_dict())
        optimizer_steps = 0

    # Prepare our optimizer
    optimizer = optimizer_function(model.parameters(), lr=learning_rate)

    for episode in range(episodes):
        state = environment.reset()

        # Convert our state to pytorch - ensure it's float
        state = torch.from_numpy(state).float()

        if render:
            environment.render()

        done = False
        step = 0
        total_reward = 0

        while not done:
            step += 1

            # Get our Q values predictions for each action
            Q = model(state)
            
            if render:
                environment.render()
            
            # Convert to numpy for ease of use
            q_values = Q.data.numpy()
            if random() < epsilon:
                # Generate a random action from the action_space of the environment
                action = environment.action_space.sample()
            else:
                # If we are not exploring, exploit the best predicted value
                # argmax returns the index location of the higest value
                action = np.argmax(q_values)

            # Take our action and 
            state2, reward, done, info = environment.step(action)

            total_reward += reward
            
            # Convert our state to pytorch - ensure it's float
            state2 = torch.from_numpy(state2).float()

            # If we are using experience replay, we now have everything we need to record
            if experience_replay:
                experience = (state, action, reward, state2, done)
                experience_memory.append(experience)

            # backpropagate is a trigger for whether or not we perform a loss
            # calculation and optimizer step. It may be skipped depending on
            # settings for the trainer
            backpropagate = False

            # If we have experience replay, we must wait until we have at least
            # a single batch of memories to train on. If we aren't, just continue
            # irregardless.
            if experience_replay and len(experience_memory) >= batch_size:
                # Mark backpropagation to trigger
                backpropagate = True

                # Create our batches from the experience memory
                experience_batch = sample(experience_memory, batch_size)
                
                # As we prepare each batch, we convert them to tensors. Since state
                # and state2 are already tensors, we use torch.cat instead of 
                # instantiating a new Tensor. We use unsqueeze here as we want the
                # tensor to be of a shape [1, <observational space>]. instead of just
                # [<observational space>] to prevent the cat from just creating a
                # long singular row of tensors
                state_batch = torch.cat([state.unsqueeze(dim=0) for (state, action, reward, state2, done) in experience_batch]) # Combine state tensors into a singular batch tensor
                action_batch = torch.Tensor([action for (state, action, reward, state2, done) in experience_batch]) # Take the sequence of actions, convert to tensor
                reward_batch = torch.Tensor([reward for (state, action, reward, state2, done) in experience_batch]) # Take the sequence of rewards, convert to tensor
                state2_batch =  torch.cat([state2.unsqueeze(dim=0) for (state, action, reward, state2, done) in experience_batch]) # Combine state2 tensors into a singular batch tensor
                done_batch = torch.Tensor([done for (state, action, reward, state2, done) in experience_batch]) # Take the sequence of done booleans, convert to tensor. This automatically onehot-encodes

                # Regenerate the Q values for the given batch at our current state
                # This is necessary because the batch may include old states from an
                # earlier, less-accurate model
                Q1 = model(state_batch)

                # Grab the expected reward future reward (Q2) for this batch
                # Turn off gradient for this batch as we aren't backpropagating on it
                # Note that we are using the target model (theta_t) instead of the
                # q model (theta_q) to prevent instability/oscillations
                with torch.no_grad():
                    # If we are using a target network, use that here.
                    if target_network:
                        Q2 = target_model(state2_batch)
                    else :
                        Q2 = model(state2_batch)

                # results is our given rewards, plus the discounted gamma of future rewards.
                # By doing (1 - done_batch), we are inverting the done_batch recording. Thus
                # we are ignoring the expected value of Q2 if our Q1 move finished the episode
                # The dimension = 1 because of the method we generated the tensor - it is columnular.
                # Or, in other words - we have a batch of N rows, each with 4 columns. We are grabbing
                # the highest value for each row. torch.max in this case returns the max value in the first
                # tensor, and the indicies in the second. We only care about the highest values, so
                # the first tensor only (hence [0])
                # Detach it from the graph - we are not backpropagating on it
                results = (reward_batch + (gamma * (1 - done_batch) * torch.max(Q2, dim=1)[0])).detach()

                # calculated is what our model expected for a reward
                #   dim=1
                #       - because we're batching, it is columnar data, same as before
                #   index=action_batch.long().unsqueeze(dim=1))
                #       - because we are selecting the value for the given action chosen.
                #           ...we must convert this to long to satisfy the function
                # .unsqueeze(dim=1)
                #   - we need to go from a shape [200] to shape [200,1] tensor
                #
                # We then call gather on the tensor, which with the above seetings and 
                # The gather function will select values from our Q1 tensor based on the
                # calculated index (which is our chosen action for the example)
                # 
                # .squeeze():
                # the gather, when done, has too many dimensions, we bring it back down to
                # tensor shape of [200]
                calculated = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

            elif not experience_replay:
                # Mark backpropagation to trigger
                backpropagate = True

                # Without saving the gradient, get the Q for the next state (Q2) as well
                with torch.no_grad():
                    # If the target network is being used, we utilize that network
                    # to get our second state. Otherwise, use the base netwwork
                    if target_network:
                        Q2 = target_model(state2)
                    else:
                        Q2 = model(state2)


                # Grab the max of the calculated next step
                maxQ = torch.max(Q2)

                # If the episode is over, return the last reward. If it is not,
                # return our reward with the maxQ predicted for the next state,
                # reduced by our gamma factor.
                if not done:
                    reward = reward + (gamma * maxQ)

                # results is our reward, separated from the graph as it's an
                # observation and we don't need to back propragate on it.
                results = torch.Tensor([reward]).detach()

                # calculated is our calculated outcome via the chosen action's
                # Q value
                calculated = Q.squeeze()[action].unsqueeze(dim=0)

            if backpropagate:
                # Irregardless of what method we used, we now have two tensors -
                # calculated and results - which represent, respectively, what
                # our Q network thinks it will get for the actions, and what it
                # actually received in that situation. We can now calculate the
                # loss

                # Calculate our loss as per the loss function. We pass it our expected
                # output, and the actual result. Here, it's the expected reward vs
                # the actual reward
                loss = loss_function(calculated, results)

                #  Zero out our gradient and backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # If we have a target network for this training, we increment our count
                # of optimizer steps. Every sync_every_steps we will copy the target
                # network weights over to the primary network.
                if target_network:
                    optimizer_steps += 1
                    if optimizer_steps % sync_every_steps == 0:
                        target_model.load_state_dict(model.state_dict())

                # Print out our progress
                print(f"Episode {episode+1} - Step {step} - Epsilon {epsilon:.4f} - Reward: {total_reward:.2f} - REWARDS - Last 100: {sum(rewards[-100:])/len(rewards[-100:]) if len(rewards) > 0 else 0:.2f} - Last 10: {sum(rewards[-10:])/len(rewards[-10:]) if len(rewards) > 0 else 0:.2f}", end="\r", flush=True)

        
            # Our state2 becomes our current state
            state = state2

        # Append the step count to steps
        steps.append(step)
        rewards.append(total_reward)

        if save_every and (episode + 1) % save_every == 0:
            model.save(f"model_episode_{episode+1}.pt")

        # After each episode, we reduce the epsilon value to slow down our exploration
        # rate. We never go below 10% for now
        if epsilon > 0.01:
            # epsilon -= 1 / episodes
            epsilon = epsilon * 0.996

        if on_episode_complete is not None:
            stop = on_episode_complete(episode, step, steps, total_reward, rewards)
            if stop:
                break

    print()

    return steps, rewards
