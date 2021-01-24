import torch
from torch import optim
from random import random, sample
import numpy as np
from collections import deque
import copy

def train(
    # Required inputs
    model, environment, loss_function, optimizer_function,
    # Standard settings
    learning_rate=1e-4, episodes=5000, gamma=0.95, render=False, device=None,
    # Epsilon greedy settings
    epsilon=1, epsilon_minimum=0.05, epsilon_minimum_at_episode=None,
    # Backpropagation settings
    batch_size=64, backpropagate_every=1,
    # Experience replay settings
    experience_replay=True, experience_memory_size=10_00,
    # Target network settings
    target_network=True, sync_every_steps=500,
    # Function controls
    state_transform=None, on_episode_complete=None, modify_reward=None,
    # Checkpoint settings
    checkpoint_model=None, checkpoint_target_model=None, checkpoint_trainer=None,
    save_every=None, save_to_folder="",
    ):

    rewards = []
    steps = []

    # If the device is not set, determine if we are going
    # to use GPU or CPU. If it is set, respect that setting
    if device == None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Calculate the epsilon decay off of epsilon_minimum and
    # epsilon_minimum_at_episode. If epsilon_minimum_at_episode
    # is None, set it to the total episode count by default
    if epsilon_minimum_at_episode is None:
        epsilon_minimum_at_episode = episodes
    # This math is derived on the idea that we want the
    # decay, over epsilon_minimum_at_episode episodes
    # to equal our epsilon minimum.
    epsilon_decay = (epsilon_minimum / epsilon) ** (1/epsilon_minimum_at_episode)


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

    # If we have a checkpoint set, we will resume from that.
    # So here we shall load up the checkpoint and 
    if checkpoint_model:
        model.load(checkpoint_model)
        if target_network:
            target_model.load(checkpoint_target_model)
        training_state = pickle.load(checkpoint_trainer)
        rewards = training_state['rewards']
        steps = training_state['steps']
        optimizer_steps = training_state['optimizer_steps']
        epsilon = training_state['epsilon']
        experience_memory = training_state['experience_memory']
        del training_state

    for episode in range(episodes):
        state_original = environment.reset()
    
        if state_transform:
            state = state_transform([state_original])
        else:
            # Convert our state to pytorch - ensure it's float
            state = torch.from_numpy(state_original).float()

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

            # Take our action and take a step
            state2_original, reward, done, info = environment.step(action)

            if modify_reward:
                reward = modify_reward(reward)

            total_reward += reward
            
            if state_transform:
                state2 = state_transform([state2_original])
            else:
                # Convert our state to pytorch - ensure it's float
                state2 = torch.from_numpy(state2_original).float()

            # If we are using experience replay, we now have everything we need to record
            if experience_replay:
                # Why are we not storing the transformed state? In the event that we are
                # utilizing a GPU, we do not want to store the experience replay memory
                # in GPU memory. We do take a hit in time for the repeated transferral,
                # but so be it for now. 
                experience = (state_original, action, reward, state2_original, done)
                experience_memory.append(experience)

            # If we have experience replay, we must wait until we have at least
            # a single batch of memories to train on. If we aren't, just continue
            # irregardless.
            if experience_replay and len(experience_memory) >= batch_size:
                # Create our batches from the experience memory
                experience_batch = sample(experience_memory, batch_size)
                
                # As we prepare each batch, we convert them to tensors. Since state
                # and state2 are already tensors, we use torch.cat instead of 
                # instantiating a new Tensor. We use unsqueeze here as we want the
                # tensor to be of a shape [1, <observational space>]. instead of just
                # [<observational space>] to prevent the cat from just creating a
                # long singular row of tensors
                if state_transform:
                    state_batch = state_transform(state_batch)
                else:
                    state_batch = torch.Tensor([state for (state, action, reward, state2, done) in experience_batch])
                action_batch = torch.Tensor([action for (state, action, reward, state2, done) in experience_batch]) # Take the sequence of actions, convert to tensor
                reward_batch = torch.Tensor([reward for (state, action, reward, state2, done) in experience_batch]) # Take the sequence of rewards, convert to tensor
                if state_transform:
                    state2_batch = state_transform(state2_batch)
                else:
                    state2_batch = torch.Tensor([state2 for (state, action, reward, state2, done) in experience_batch])
                done_batch = torch.Tensor([done for (state, action, reward, state2, done) in experience_batch]) # Take the sequence of done booleans, convert to tensor. This automatically onehot-encodes

                # This is done in case the state transformation function fails to.
                # It should be a no-op in the event that the tensors were already
                # on the disk.
                state_batch = state_batch.to(device)
                action_batch = action_batch.to(device)
                reward_batch = reward_batch.to(device)
                state2_batch = state2_batch.to(device)
                done_batch = done_batch.to(device)

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

            if (not experience_replay or (experience_replay and len(experience_memory) >= batch_size)) \
                and step % backpropagate_every == 0:
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

            # Our state2 becomes our current state
            # We also transfer
            state = state2
            state_original = state2_original

        # Append the step count to steps
        steps.append(step)
        rewards.append(total_reward)

        if save_every and (episode + 1) % save_every == 0:
            model.save(f"model_episode_{episode+1}.pt")

        # After each episode, we reduce the epsilon value to slow down our exploration
        # rate. We never go below epsilon_minimum.
        if epsilon > epsilon_minimum:
            epsilon *= epsilon_decay
        else:
            epsilon = epsilon_minimum

        # If our save_every triggers, saved the model, our target model (if used), and
        # what traininer state/variables we need to continue on from this point.
        # Note that this process tends to be slow
        if save_every and (episode + 1) % save_every == 0:
            model.save(f"{save_to_folder}/model_episode_{episode+1}.pt")
            if target_network:
                target_model.save(f"{save_to_folder}/target_network_episode_{episode+1}.pt")
            training_state = {
                "experience_memory": experience_memory,
                "rewards": rewards,
                "steps": steps,
                "epsilon": epsilon,   
                "optimizer_steps": optimizer_steps,
            }
            pickle.dump(training_state, open(f"{save_to_folder}/training_state_{episode+1}.pt", "wb"))

        if on_episode_complete:
            stop = on_episode_complete(episode, step, steps, total_reward, rewards)
            if stop:
                break

    return steps, rewards
