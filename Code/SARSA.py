
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions

'''
Chapter 3. SARSA
Most code here is copied from SLM-Lab first and then modified to show a plain torch implementation.
'''

# This is a modified Categorical distribution class to implement greedy policy.
class Argmax(distributions.Categorical):
    '''
    Special distribution class for argmax sampling, where probability is always 1 for the argmax.
    NOTE although argmax is not a sampling distribution, this implementation is for API consistency.
    '''
    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            new_probs = torch.zeros_like(probs, dtype=torch.float)
            new_probs[probs == probs.max(dim=-1, keepdim=True)[0]] = 1.0
            probs = new_probs
        elif logits is not None:
            new_logits = torch.full_like(logits, -1e8, dtype=torch.float)
            new_logits[logits == logits.max(dim=-1, keepdim=True)[0]] = 1.0
            logits = new_logits

        super().__init__(probs=probs, logits=logits, validate_args=validate_args)

class SARSA(nn.Module):
    def __init__(self, env):
        super(SARSA, self).__init__()

        self.env = env
        in_dim = env.observation_space.shape[0] # 4 for CartPole
        out_dim = env.action_space.n # 2 for CardPole

        # Initialize the neural network used to learn the Q function
        # Let's use a single hidden layer with 64 units and SELU as an activation function.
        layers = [
            nn.Linear(in_dim, 64),
            nn.SELU(),
            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.train()

        # Optimizer - RMSprop
        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.01)

        # Gamma
        self.gamma = 0.99

        # Memory for batch
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.memory = {k: [] for k in self.data_keys}

        # Epsilon greedy policy
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_max_steps = 10000
        self.epsilon = self.epsilon_start
        self.current_step = 0

        # etc.
        self.to_train = 0
        self.training_frequency = 32 # number of experiences to collect before training.

    def act(self, state):
        state = torch.from_numpy(state.astype(np.float32))

        action = None

        # Epsilon greedy to balance between exploring and exploiting.
        if self.epsilon > np.random.rand():
            action = self.random_policy()
        else:
            action = self.greedy_policy(state)

        return action.item()

    def random_policy(self):
        action = [self.env.action_space.sample()]
        return torch.tensor(action)

    def greedy_policy(self, state):
        pdparam = self.model(state)
        action_pd = Argmax(logits=pdparam)  
        return action_pd.sample()

    def sample(self):
        # Create batch
        batch = {k: self.memory[k] for k in self.data_keys}

        # 'next_actions' is copied from 'actions' from index 1 and its last element will be always 0.
        # This is safe for next_action at done since the calculated act_next_q_preds will be multiplied by (1 - batch['dones'])
        batch['next_actions'] = np.zeros_like(batch['actions'])
        batch['next_actions'][:-1] = batch['actions'][1:]

        for k in batch:
            batch[k] = np.array(batch[k])
            batch[k] = torch.from_numpy(batch[k].astype(np.float32))

        return batch

    def calc_q_loss(self, batch):
        states = batch['states']
        next_states = batch['next_states']

        q_preds = self.model(states)
        with torch.no_grad():
            next_q_preds = self.model(next_states)

        """
        This is the gut of SARSA implementation.
            : Q-value = Q(state, action) : NN(state) generates Q-values for all actions.

        We treat that our NN (self.model) generates Q-values for each action. For example, 
        q_preds are just logits from NN and we assume logits[0] is the Q-value for action 0.
        Therefore, act_q_preds should be the Q-value (logit) of the action selected for the state.
        We can do this by selecting one of logits(q_preds) by using 'action' as an index.
        torch.gather(torch.tensor.gather) exactly does that.
        """

        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        act_next_q_preds = next_q_preds.gather(-1, batch['next_actions'].long().unsqueeze(-1)).squeeze(-1)
        act_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * act_next_q_preds

        #print(f'act_q_preds: {act_q_preds}\nact_q_targets: {act_q_targets}')

        """
        You can now easily understand that SARSA is on-policy RL algorithm that cannot reuse
        experience produced by a different policy (i.e NN whose parameters are different).
        This is because the loss is only meaningful to update NN when act_q_preds and act_q_targets
        are from the same NN. It makes no sense to update NN parameters to reduce this difference
        when act_q_preds and act_q_targets are came from different NNs.
        """

        # Let's use mean-squared-error loss function.
        loss = nn.MSELoss()
        q_loss = loss(act_q_preds, act_q_targets)
        return q_loss

    def check_train(self):
        if self.to_train == 1:

            # Computer loss for the batch.
            batch = self.sample()
            loss = self.calc_q_loss(batch)

            # Computer gradients with backpropagation.
            self.optim.zero_grad()
            loss.backward()

            # Update NN parameters.
            self.optim.step()

            # Reset
            self.to_train = 0
            self.memory = {k: [] for k in self.data_keys}

    def update_memory(self, state, action, reward, next_state, done):
        # Add this exp to memory.
        most_recent = (state, action, reward, next_state, done)
        for idx, k in enumerate(self.data_keys):
            self.memory[k].append(most_recent[idx])

        # If it has collected the desired number of experiences, it is ready to train.
        if len(self.memory['states']) == self.training_frequency:
            self.to_train = 1

    def update_epsilon(self):
        # Simple linear decay
        if self.epsilon_max_steps <= self.current_step:
            self.epsilon = self.epsilon_end
            return

        slope = (self.epsilon_end - self.epsilon_start) / (self.epsilon_max_steps - self.current_step)
        self.epsilon = max(slope*self.current_step + self.epsilon_start, self.epsilon_end)

    def update(self, state, action, reward, next_state, done):
        self.current_step += 1
        self.update_memory(state, action, reward, next_state, done)
        self.check_train()
        self.update_epsilon()

def run_rl(sarsa, env, max_frame):
    state = env.reset()
    done = False
    cur_frame = 0;
    total_reward = 0
    while True:
        if done:  # before starting another episode
            print(f'Episode done: cur_frame={cur_frame} total_reward={total_reward}')
            total_reward = 0

            if cur_frame < max_frame:  # reset and continue
                state = env.reset()
                done = False

        if cur_frame >= max_frame:  # finish
            break

        cur_frame += 1

        action = sarsa.act(state)
        next_state, reward, done, info = env.step(action)
        sarsa.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

def main():
    env = gym.make("CartPole-v0")
    sarsa = SARSA(env)

    run_rl(sarsa, env, max_frame=100000)

if __name__ == '__main__':
    main()

   

