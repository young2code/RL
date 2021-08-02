
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions
import random

'''
Chapter 5. Improving DQN
Most code here is copied from SLM-Lab first and then modified to show a plain torch implementation.

This is for introducing PER(Prioritized Experience Replay) to choose most beniefical samples to train.
'''

class SumTree:
    '''
    Helper class for PrioritizedReplay

    This implementation is, with minor adaptations, Jaromír Janisch's. The license is reproduced below.
    For more information see his excellent blog series "Let's make a DQN" https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/

    MIT License

    Copyright (c) 2018 Jaromír Janisch

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    '''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores the priorities and sums of priorities
        self.indices = np.zeros(capacity)  # Stores the indices of the experiences

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, index):
        idx = self.write + self.capacity - 1

        self.indices[self.write] = index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        indexIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.indices[indexIdx])

    def print_tree(self):
        for i in range(len(self.indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Data idx: {self.indices[i]}, Prio: {self.tree[j]}')


class DoubleDQN_PER(nn.Module):
    def __init__(self, env):
        super(DoubleDQN_PER, self).__init__()

        self.env = env
        in_dim = env.observation_space.shape[0] # 4 for CartPole
        out_dim = env.action_space.n # 2 for CardPole

        # Initialize the neural network used to learn the Q function
        layers = [
            nn.Linear(in_dim, 64),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.Linear(32, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.target_model = nn.Sequential(*layers)
        self.model_update_frequency = 1000
        self.train()

        # Optimizer - Adam with learning rate linear decay.
        self.learning_rate = 0.01
        self.learning_rate_max_steps = 10000
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, 
                                                              lr_lambda=lambda x: 1 - x/self.learning_rate_max_steps if x < self.learning_rate_max_steps else 1/self.learning_rate_max_steps)
        # Gamma
        self.gamma = 0.99

        # Memory for batch
        # adds a 'priorities' scalar to the data_keys for PER.
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities']
        self.memory_batch_size = 32
        self.memory_max_size = 10000
        self.memory_cur_size = 0
        self.memory_seen_size = 0
        self.memory_head = -1
        self.memory = {k: [None] * self.memory_max_size for k in self.data_keys}

        # PER
        self.batch_idxs = None
        self.tree_idxs = None
        self.alpha = 0.6 # 
        self.epsilon = 0.0001 # a small positive number that prevent sexperiences from never being sampled.
        self.tree = SumTree(self.memory_max_size)

        # Boltzmann policy
        self.boltzmann_tau_start = 5.0
        self.boltzmann_tau_end = 0.5
        self.boltzmann_tau_max_steps = 10000
        self.boltzmann_tau = self.boltzmann_tau_start

        # Training with Replay Experiences
        self.to_train = 0
        self.training_batch_iter = 8
        self.training_iter = 4
        self.training_frequency = 4
        self.training_start_step = 32
        self.current_training_step = 0

        # Frame
        self.current_frame = 0
        self.max_frame = 10000

    def act(self, state):
        state = torch.from_numpy(state.astype(np.float32))
        action = self.boltzmann_policy(state)
        return action.item()

    def boltzmann_policy(self, state):
        '''
        Boltzmann policy: adjust pdparam with temperature tau; 
        the higher the more randomness/noise in action.
        '''
        pdparam = self.model(state)
        pdparam /= self.boltzmann_tau
        action_pd = distributions.Categorical(logits=pdparam)
        return action_pd.sample()

    def sample_idxs(self):
        '''Samples batch_size indices from memory in proportional to their priority.'''
        batch_idxs = np.zeros(self.memory_batch_size)
        tree_idxs = np.zeros(self.memory_batch_size, dtype=int)

        for i in range(self.memory_batch_size):
            s = random.uniform(0, self.tree.total())
            (tree_idx, p, idx) = self.tree.get(s)
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx

        self.batch_idxs = np.asarray(batch_idxs).astype(int)
        self.tree_idxs = tree_idxs

    def sample(self):

        self.sample_idxs()

        # Create batch.
        batch = {k: [] for k in self.data_keys}
        for index in self.batch_idxs:
            for k in self.data_keys:
                batch[k].append(self.memory[k][index])

        for k in batch:
            batch[k] = np.array(batch[k])
            batch[k] = torch.from_numpy(batch[k].astype(np.float32))

        return batch

    def calc_q_loss(self, batch):
        states = batch['states']
        next_states = batch['next_states']

        q_preds = self.model(states)

        """
        This is the gut of Double DQN implementation.
            : We use two different models (networks) to reduce the overestimation of Q-value.
            : One for selecting the Q-maximizing action, a`, and another for the Q value of a` given next state, s`.
            : self.model is representing the first model and self.target_model is the second model here.
            :
            : The second model also acts as a target network that helps with stabilizing learning by reducing
            : a moving target issue (which is Q_target that keeps changing if a single model (network) is used).
        """

        with torch.no_grad():
            online_next_q_preds = self.model(next_states)
            next_q_preds = self.target_model(next_states)

        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)

        """
        For SARSA, we calculate Q-values for the next action taken during the episode like the below.

            act_next_q_preds = next_q_preds.gather(-1, batch['next_actions'].long().unsqueeze(-1)).squeeze(-1)        

        For DQN, it assumes there is a perfect policy and the next action should be always the best.
        Thus, instead of taking the Q value of the next action, it just takes the maximum Q-value for
        the next state. This is why DQN is off-policy RL algorithm since it does not rely on the current
        policy (that is used to choose next action) while training.

            max_next_q_preds, _ = next_q_preds.max(dim=-1, keepdim=False)

        For Double DQN, we use the first model (self.model) to choose the Q-maximizing action (online_action).
        Then use the second model (self.target_model) to get the Q value of a` and s`.
        """

        online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)
        max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)

        act_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * max_next_q_preds

        #print(f'act_q_preds: {act_q_preds}\nmax_next_q_preds: {max_next_q_preds}')

        # Let's use mean-squared-error loss function.
        loss = nn.MSELoss()
        q_loss = loss(act_q_preds, act_q_targets)

        # PER: Update priorities of these batch samples with q estimation differences.
        errors = (act_q_targets - act_q_preds.detach()).abs().cpu().numpy()
        self.update_priorities(errors)

        return q_loss

    def check_train(self):
        if self.to_train == 1:

            for _ in range(self.training_iter):
                batch = self.sample()

                for _ in range(self.training_batch_iter):
                    # Compute loss for the batch.
                    loss = self.calc_q_loss(batch)

                    # Compute gradients with backpropagation.
                    self.optim.zero_grad()
                    loss.backward()

                    # Update NN parameters.
                    self.optim.step()

                    self.current_training_step += 1

            # Reset
            self.to_train = 0

    def get_priority(self, error):
        '''Takes in the error of one or more examples and returns the proportional priority'''
        return np.power(error + self.epsilon, self.alpha).squeeze()

    def update_priorities(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        priorities = self.get_priority(errors)
        assert len(priorities) == self.batch_idxs.size
        for idx, p in zip(self.batch_idxs, priorities):
            self.memory['priorities'][idx] = p
        for p, i in zip(priorities, self.tree_idxs):
            self.tree.update(i, p)

    def update_memory(self, state, action, reward, next_state, done, error=100000):
        """
        Add this exp to memory. Since DQN is off-policy algorithm, we can reuse
        any experiences generated during training regardless of which policy (NN)
        is used. We will discard the oldest exp if there is no space to add new one.
        """

        priority = self.get_priority(error)
        most_recent = (state, action, reward, next_state, done, priority)
        self.memory_head = (self.memory_head + 1) % self.memory_max_size

        for idx, k in enumerate(self.data_keys):
            self.memory[k][self.memory_head] = most_recent[idx]

        self.tree.add(priority, self.memory_head)

        if self.memory_cur_size < self.memory_max_size:
            self.memory_cur_size += 1

        self.memory_seen_size += 1

        self.to_train = self.memory_seen_size > self.training_start_step and self.memory_head % self.training_frequency == 0;

    def update_tau(self):
        # Simple linear decay
        if self.boltzmann_tau_max_steps <= self.current_frame:
            self.boltzmann_tau = self.boltzmann_tau_end
            return

        slope = (self.boltzmann_tau_end - self.boltzmann_tau_start) / (self.boltzmann_tau_max_steps - self.current_frame)
        self.boltzmann_tau = max(slope*self.current_frame + self.boltzmann_tau_start, self.boltzmann_tau_end)

    def update_models(self):
        if self.current_frame % self.model_update_frequency == 0:
            # replace
            self.target_model.load_state_dict(self.model.state_dict())

    def update(self, state, action, reward, next_state, done):
        self.current_frame += 1
        self.update_memory(state, action, reward, next_state, done)
        self.check_train()
        self.update_tau()
        self.update_models()
        if (self.memory_seen_size > self.training_start_step):
            self.lr_scheduler.step()

def run_rl(dqn, env):
    state = env.reset()
    done = False
    total_reward = 0
    while True:
        if done:  # before starting another episode
            print(f'Episode done: cur_frame={dqn.current_frame} current_training_step={dqn.current_training_step} total_reward={total_reward}')
            total_reward = 0

            if dqn.current_frame < dqn.max_frame:  # reset and continue
                state = env.reset()
                done = False

        if dqn.current_frame >= dqn.max_frame:  # finish
            break

        action = dqn.act(state)
        next_state, reward, done, info = env.step(action)
        dqn.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

def main():
    env = gym.make("CartPole-v0")
    dqn = DoubleDQN_PER(env)
    run_rl(dqn, env)

if __name__ == '__main__':
    main()