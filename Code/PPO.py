
from copy import deepcopy
from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions
import random

'''
Chapter 7. Proximal Policy Optimization (PPO)
Most code here is copied from SLM-Lab first and then modified to show a plain torch implementation.

For this example, we are implementing PPO with clipped surrogate objective with two seperate networks for actor and critic.
'''

class PPO(nn.Module):
    def __init__(self, env):
        super(PPO, self).__init__()

        self.env = env
        in_dim = env.observation_space.shape[0] # 4 for CartPole
        out_dim = env.action_space.n # 2 for CardPole

        # Initialize the neural networks for Actor and Critic
        # We do not share NN between actor and critic for this example.

        # Actor
        actor_layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]
        self.actor_model = nn.Sequential(*actor_layers)
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=0.01)
        self.actor_policy_loss_coef = 1.0
        self.actor_entropy_coef = 0.001

        # Critic
        critic_layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ]
        self.critic_model = nn.Sequential(*critic_layers)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr=0.01)
        self.critic_val_loss_coef = 1.0

        # Create old net to calculate ratio
        self.old_actor_model = deepcopy(self.actor_model)

        self.train()

        # Gamma
        self.gamma = 0.99

        # Memory for batch
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.memory = {k: [] for k in self.data_keys}

        # PPO uses GAE
        self.gae_lambda = 0.95
        self.clip_eps = 0.2

        # Training
        self.to_train = 0
        self.training_frequency = 32
        self.training_epoch = 4

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


    def calc_gaes(self, rewards, dones, v_preds):
        '''
        Estimate the advantages using GAE from Schulman et al. https://arxiv.org/pdf/1506.02438.pdf
        v_preds are values predicted for current states, with one last element as the final next_state
        delta is defined as r + gamma * V(s') - V(s) in eqn 10
        GAE is defined in eqn 16
        This method computes in torch tensor to prevent unnecessary moves between devices (e.g. GPU tensor to CPU numpy)
        NOTE any standardization is done outside of this method
        '''
        T = len(rewards)
        assert T + 1 == len(v_preds), f'T+1: {T+1} v.s. v_preds.shape: {v_preds.shape}'  # v_preds runs into t+1
        gaes = torch.zeros_like(rewards)
        future_gae = torch.tensor(0.0, dtype=rewards.dtype)
        not_dones = 1 - dones  # to reset at episode boundary by multiplying 0
        deltas = rewards + self.gamma * v_preds[1:] * not_dones - v_preds[:-1]
        coef = self.gamma * self.gae_lambda
        for t in reversed(range(T)):
            gaes[t] = future_gae = deltas[t] + coef * not_dones[t] * future_gae
        return gaes

    def calc_gae_advs_v_targets(self, batch, v_preds):
        '''
        Calculate GAE, and advs = GAE, v_targets = advs + v_preds
        See GAE from Schulman et al. https://arxiv.org/pdf/1506.02438.pdf
        '''
        next_states = batch['next_states'][-1]
        next_states = next_states.unsqueeze(dim=0)

        with torch.no_grad():
            next_v_pred = self.calc_v(next_states)

        v_preds = v_preds.detach()  # adv does not accumulate grad
        v_preds_all = torch.cat((v_preds, next_v_pred), dim=0)
        advs = self.calc_gaes(batch['rewards'], batch['dones'], v_preds_all)
        v_targets = advs + v_preds            
        advs = (advs - advs.mean()) / (advs.std() + 1e-08)  # standardize only for advs, not v_targets

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

    def calc_policy_loss(self, batch, pdparams, advs):
        '''
        The PPO loss function (subscript t is omitted)
        L^{CLIP+VF+S} = E[ L^CLIP - c1 * L^VF + c2 * H[pi](s) ]

        Breakdown piecewise,
        1. L^CLIP = E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]
        where ratio = pi(a|s) / pi_old(a|s)

        2. L^VF = E[ mse(V(s_t), V^target) ]

        3. H = E[ entropy ]
        '''

        action_pd = Categorical(logits=pdparams) # probability distribution
        states = batch['states']
        actions = batch['actions']

        # L^CLIP
        log_probs = action_pd.log_prob(actions)
        with torch.no_grad():
            old_pdparams = self.old_actor_model(states)
            old_action_pd = Categorical(logits=old_pdparams) # probability distribution
            old_log_probs = old_action_pd.log_prob(actions)
        assert log_probs.shape == old_log_probs.shape
        ratios = torch.exp(log_probs - old_log_probs)
        #print(f'ratios: {ratios}')
        sur_1 = ratios * advs
        sur_2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
        # flip sign because need to maximize
        clip_loss = -torch.min(sur_1, sur_2).mean()
        #print(f'clip_loss: {clip_loss}')

        # L^VF (inherit from ActorCritic)

        # H entropy regularization
        entropy = action_pd.entropy().mean()
        ent_penalty = -self.actor_entropy_coef * entropy
        #print(f'ent_penalty: {ent_penalty}')

        policy_loss = clip_loss + ent_penalty
        #print(f'PPO Actor policy loss: {policy_loss:g}')
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
            # update old net
            self.old_actor_model.load_state_dict(self.actor_model.state_dict()) 

            batch = self.sample()
            pdparams, v_preds = self.calc_pdparam_v(batch)
            advs, v_targets = self.calc_gae_advs_v_targets(batch, v_preds)

            for _ in range(self.training_epoch):
                pdparams, v_preds = self.calc_pdparam_v(batch)
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

            # reset
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
    ppo = PPO(env)
    run_rl(ppo, env, max_frame=1000)

if __name__ == '__main__':
    main()