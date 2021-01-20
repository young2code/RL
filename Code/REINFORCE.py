from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

'''
Chapter 2. REINFORCE
Most code here is copied from SLM-Lab first and then modified to show a plain torch implementation.
'''

gamma = 0.99

# Policy Pi
class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()

        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]

        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        
        """
        - Action probability distribution = Policy(state) : NN(state) generates probabilities for all actions.
        - They are actually just logits which are not normalized, unlike probabilities that sum up to 1.
        - Categorical() will sample action based on these logits by using Softmax.
        - Softmax - https://miro.medium.com/max/875/1*ReYpdIZ3ZSAPb2W8cJpkBg.jpeg
        - Categorical() also provides log(action_probability) that we need for calculating loss.
        """

        x = torch.from_numpy(state.astype(np.float32)) # to tensor
        pdparam = self.forward(x) # forward pass

        pd = Categorical(logits=pdparam) # probability distribution
        action = pd.sample() # pi(a|s) in action via pd
        log_prob = pd.log_prob(action) # log_prob prob of pi(a|s)
        self.log_probs.append(log_prob)
        return action.item()

def train(pi, optimizer):
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32) # the returns
    future_ret = 0.0

    # Compute the discounted returns efficiently in a reversed order.
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret

    # Compute loss (which is really opposite of reward)
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets # gradient term: Negative for maximizing reward
    loss = torch.sum(loss)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward() # backpropagate, compute gradients that will be stored by the tensors (parameters)
    optimizer.step() # gradient-ascent, update the weights

    return loss

def main():
    env = gym.make("CartPole-v0")
    in_dim = env.observation_space.shape[0] # 4
    out_dim = env.action_space.n # 2
    pi = Pi(in_dim, out_dim)
    optimizer = optim.Adam(pi.parameters(), lr=0.01)

    for epi in range(300):
        state = env.reset()

        for t in range(200): # cartpole max timestep is 200
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            # env.render()

            if done:
                break

        loss = train(pi, optimizer)
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()
        print(f"Episode {epi}, loss: {loss}, total_reward: {total_reward}, solved: {solved}")

if __name__ == '__main__':
    main()
