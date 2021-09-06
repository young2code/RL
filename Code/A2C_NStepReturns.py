
from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions
import random

'''
Chapter 6. Advantage Actor-Critic (A2C)
Most code here is copied from SLM-Lab first and then modified to show a plain torch implementation.

This exmaple will create two seperate networs for actor and critic. For advantage estimation,
it will use N-step returns method.
'''

class A2C(nn.Module):
    def __init__(self, env):
        super(A2C, self).__init__()

        self.env = env
        in_dim = env.observation_space.shape[0] # 4 for CartPole
        out_dim = env.action_space.n # 2 for CardPole

        # Initialize the neural networks for Actor and Critic
        # We do not share NN between actor and critic for this example.

        # Actor
        actor_layers = [
            nn.Linear(in_dim, 64),
            nn.SELU(),
            nn.Linear(64, out_dim),
        ]
        self.actor_model = nn.Sequential(*actor_layers)
        self.actor_optim = torch.optim.RMSprop(self.actor_model.parameters(), lr=0.01)
        self.actor_policy_loss_coef = 1.0
        self.actor_entropy_coef = 0.01

        # Critic
        critic_layers = [
            nn.Linear(in_dim, 64),
            nn.SELU(),
            nn.Linear(64, 1),
        ]
        self.critic_model = nn.Sequential(*critic_layers)
        self.critic_optim = torch.optim.RMSprop(self.critic_model.parameters(), lr=0.01)
        self.critic_val_loss_coef = 1.0

        self.train()

        # Gamma
        self.gamma = 0.99

        # Memory for batch
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.memory = {k: [] for k in self.data_keys}

        # Training
        self.to_train = 0
        self.num_step_returns = 32
        self.training_frequency = self.num_step_returns        

    def act(self, state):      
        """
        - Probability distribution = Policy(state) : NN(state) generates v value for all actions given a state.
        - They are actually just logits which are not normalized, unlike probabilities that sum up to 1.
        - Categorical() will sample action based on these logits by using Softmax.
        - Softmax - https://miro.medium.com/max/875/1*ReYpdIZ3ZSAPb2W8cJpkBg.jpeg
        - Categorical() also provides log(action_probability) that we need for calculating loss.
        """
        x = torch.from_numpy(state.astype(np.float32)) # to tensor        
        pdparam = self.actor_model(x) # forward pass

        pd = Categorical(logits=pdparam) # probability distribution
        action = pd.sample() # pi(a|s) in action via pd
        return action.item()

    def sample(self):
        # Create batch
        batch = {k: self.memory[k] for k in self.data_keys}

        for k in batch:
            batch[k] = np.array(batch[k])
            batch[k] = torch.from_numpy(batch[k].astype(np.float32))

        return batch

    def update_memory(self, state, action, reward, next_state, done):
        # Add this exp to memory.
        most_recent = (state, action, reward, next_state, done)
        for idx, k in enumerate(self.data_keys):
            self.memory[k].append(most_recent[idx])

        # If it has collected the desired number of experiences, it is ready to train.
        if len(self.memory['states']) == self.training_frequency:
            self.to_train = 1

    def calc_v(self, x):
        '''
        Forward-pass to calculate the predicted state-value from critic_net.
        '''
        return self.critic_model(x).view(-1)

    def calc_pdparam_v(self, batch):
        '''Efficiently forward to get pdparam and v by batch for loss computation'''
        states = batch['states']
        pdparam = self.actor_model(states)
        v_pred = self.calc_v(states)
        return pdparam, v_pred

    def calc_nstep_returns(self, batch, next_v_pred):
        '''
        Estimate the advantages using n-step returns. Ref: http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%207.pdf
        Also see Algorithm S3 from A3C paper https://arxiv.org/pdf/1602.01783.pdf for the calculation used below
        R^(n)_t = r_{t} + gamma r_{t+1} + ... + gamma^(n-1) r_{t+n-1} + gamma^(n) V(s_{t+n})
        This is how we estimate q value (s,a) with rewards and v value (s).
        '''
        rewards = batch['rewards']
        dones = batch['dones']
        rets = torch.zeros_like(rewards)
        future_ret = next_v_pred
        not_dones = 1 - dones

        for t in reversed(range(self.num_step_returns)):
            rets[t] = future_ret = rewards[t] + self.gamma * future_ret * not_dones[t]

        return rets

    def calc_nstep_advs_v_targets(self, batch, v_preds):
        '''
        Calculate N-step returns, and advs = nstep_rets - v_preds, v_targets = nstep_rets
        See n-step advantage under http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf
        '''
        next_states = batch['next_states'][-1]

        with torch.no_grad():
            next_v_pred = self.calc_v(next_states)

        v_preds = v_preds.detach()  # adv does not accumulate grad
        nstep_rets = self.calc_nstep_returns(batch, next_v_pred)
        advs = nstep_rets - v_preds
        v_targets = nstep_rets

        #print(f'advs: {advs}\nv_targets: {v_targets}')

        return advs, v_targets

    def calc_policy_loss(self, batch, pdparams, advs):
        '''Calculate the actor's policy loss'''        
        action_pd = Categorical(logits=pdparams) # probability distribution
        actions = batch['actions']
        log_probs = action_pd.log_prob(actions)
        policy_loss = -self.actor_policy_loss_coef * (log_probs * advs).mean()

        # Entropy Regularization
        entropy = action_pd.entropy().mean()
        policy_loss += (-self.actor_entropy_coef * entropy)

        #print(f'Actor policy loss: {policy_loss:g}')
        return policy_loss

    def calc_val_loss(self, v_preds, v_targets):
        '''Calculate the critic's value loss'''
        assert v_preds.shape == v_targets.shape, f'{v_preds.shape} != {v_targets.shape}'

        # Let's use mean-squared-error loss function.
        loss = nn.MSELoss()
        val_loss = self.critic_val_loss_coef * loss(v_preds, v_targets)
        #print(f'Critic value loss: {val_loss:g}')
        return val_loss

    def check_train(self):
        if self.to_train == 1:
            batch = self.sample()
            pdparams, v_preds = self.calc_pdparam_v(batch)
            advs, v_targets = self.calc_nstep_advs_v_targets(batch, v_preds)
            policy_loss = self.calc_policy_loss(batch, pdparams, advs)  # from actor
            val_loss = self.calc_val_loss(v_preds, v_targets)  # from critic

            # actor update
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            # critic update
            self.critic_optim.zero_grad()
            val_loss.backward()
            self.critic_optim.step()

            # Reset- A2C is an on-policy algorithm so we cannot reuse data for next training.
            self.to_train = 0
            self.memory = {k: [] for k in self.data_keys}
            

    def update(self, state, action, reward, next_state, done):
        self.update_memory(state, action, reward, next_state, done)
        self.check_train()


def run_rl(a2c, env, max_frame):
    state = env.reset()
    done = False
    total_reward = 0
    current_frame = 0
    total_reward = 0

    while True:
        if done:  # before starting another episode
            print(f'Episode done: cur_frame={current_frame} total_reward={total_reward}')
            total_reward = 0

            if current_frame < max_frame:  # reset and continue
                state = env.reset()
                done = False

        if current_frame >= max_frame:  # finish
            break

        action = a2c.act(state)
        next_state, reward, done, info = env.step(action)
        a2c.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

def main():
    env = gym.make("CartPole-v0")
    a2c = A2C(env)
    run_rl(a2c, env, max_frame=1000)

if __name__ == '__main__':
    main()