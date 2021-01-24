
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions

'''
Chapter 4. DQN
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

class DQN(nn.Module):
    def __init__(self, env):
        super(DQN, self).__init__()

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
        self.train()

        # Optimizer - Adam
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Gamma
        self.gamma = 0.99

        # Memory for batch
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.memory_batch_size = 32
        self.memory_max_size = 10000
        self.memory_cur_size = 0
        self.memory_seen_size = 0
        self.memory_head = -1
        self.memory = {k: [None] * self.memory_max_size for k in self.data_keys}

        # Need to use boltzmann instead
        # Epsilon greedy policy
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_max_steps = 10000
        self.epsilon = self.epsilon_start
        self.current_step = 0

        # Learning rate decay to zero
        self.learning_rate_max_steps = 30000

        # Training
        self.to_train = 0
        self.training_batch_iter = 8
        self.training_iter = 4
        self.training_frequency = 4
        self.training_start_step = 32

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
        # Batch indices a sampled random uniformly among experiences in memory.
        batch_idxs = np.random.randint(self.memory_cur_size, size=self.memory_batch_size)

        # Create batch.
        batch = {k: [] for k in self.data_keys}
        for index in batch_idxs:
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
        with torch.no_grad():
            next_q_preds = self.model(next_states)

        """
        This is the gut of DQN implementation.
            : Q-value = Q(state, action) : NN(state) generates Q-values for all actions.

        We treat that our NN (self.model) generates Q-values for each action. For example, 
        q_preds are just logits from NN and we assume logits[0] is the Q-value for action 0.
        Therefore, act_q_preds should be the Q-value (logit) of the action selected for the state.
        We can do this by selecting one of logits(q_preds) by using 'action' as an index.
        torch.gather(torch.tensor.gather) exactly does that.
        """

        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)

        """
        For SARSA, we calculate Q-values for the next action taken during the episode like the below.
        act_next_q_preds = next_q_preds.gather(-1, batch['next_actions'].long().unsqueeze(-1)).squeeze(-1)        

        For DQN, it assumes there is a perfect policy and the next action should be always the best.
        Thus, instead of taking the Q value of the next action, it just takes the maximum Q-value for
        the next state. This is why DQN is off-policy RL algorithm since it does not rely on the current
        policy (that is used to choose next action) while training.
        """

        max_next_q_preds, _ = next_q_preds.max(dim=-1, keepdim=False)
        act_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * max_next_q_preds

        #print(f'act_q_preds: {act_q_preds}\nmax_next_q_preds: {max_next_q_preds}')

        # Let's use mean-squared-error loss function.
        loss = nn.MSELoss()
        q_loss = loss(act_q_preds, act_q_targets)
        return q_loss

    def check_train(self):
        if self.to_train == 1:

            for _ in range(self.training_iter):
                batch = self.sample()

                for _ in range(self.training_batch_iter):
                    # Computer loss for the batch.
                    loss = self.calc_q_loss(batch)

                    # Computer gradients with backpropagation.
                    self.optim.zero_grad()
                    loss.backward()

                    # Update NN parameters.
                    self.optim.step()

            # Reset
            self.to_train = 0


    def update_memory(self, state, action, reward, next_state, done):
        """
        Add this exp to memory. Since DQN is off-policy algorithm, we can reuse
        any experiences generated during training regardless of which policy (NN)
        is used. We will discard the oldest exp if there is no space to add new one.
        """

        most_recent = (state, action, reward, next_state, done)
        self.memory_head = (self.memory_head + 1) % self.memory_max_size

        for idx, k in enumerate(self.data_keys):
            self.memory[k][self.memory_head] = most_recent[idx]

        if self.memory_cur_size < self.memory_max_size:
            self.memory_cur_size += 1

        self.memory_seen_size += 1

        self.to_train = self.memory_seen_size > self.training_start_step and self.memory_head % self.training_frequency == 0;

    def update_epsilon(self):
        # Simple linear decay
        self.current_step += 1

        if self.epsilon_max_steps <= self.current_step:
            self.epsilon = self.epsilon_end
            return

        slope = (self.epsilon_end - self.epsilon_start) / (self.epsilon_max_steps - self.current_step)
        self.epsilon = max(slope*self.current_step + self.epsilon_start, self.epsilon_end)

    def update(self, state, action, reward, next_state, done):
        self.update_memory(state, action, reward, next_state, done)
        self.check_train()
        self.update_epsilon()

def run_rl(dqn, env, max_frame):
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

        action = dqn.act(state)
        next_state, reward, done, info = env.step(action)
        dqn.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

def main():
    env = gym.make("CartPole-v0")
    dqn = DQN(env)

    run_rl(dqn, env, max_frame=30000)

if __name__ == '__main__':
    main()

   

