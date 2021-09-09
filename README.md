# RL
Reinforcement Learning Algorithms

# Background
I've implemented all the RL algorithms introduced in this book - ["Foundations Of Deep Reinforcement Learning"](https://www.amazon.com/gp/product/0135172381). They are REINFORCE, SARSA, DQN, A2C and PPO.

![alt text](https://images-na.ssl-images-amazon.com/images/I/41HraVa1zgS._SX218_BO1,204,203,200_QL40_FMwebp_.jpg)

While the book is really awesome, its code examples are not easy to read as they are implemented as a part of this big RL framework called [SLM-Lab](https://slm-lab.gitbook.io/slm-lab/) except for the very first algorithm, REINFORCE. Therefore, I've decided to write simple and easy to understand code that shows each algorithm's core element clearly. Most of the code lines here are copied from https://github.com/kengz/SLM-Lab and modified so that each file contains a complete single algorithm without relying on any other files. I've also add some more comments on parts that I was confused. I hope it helps anyone studying the book or those RL algorithms. :)

# Install
Please install the below two packages to run.

- PyTorch
https://pytorch.org/get-started/locally/

```
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- Gym
https://gym.openai.com/docs/

```
pip install gym
```
# Run
You should be able to run each python file representing one RL algorithm. I used [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) as a problem to solve and verified that all the algorithms can solve it by reaching 200 as a total reward. Here is an example output when running [DoubleDQN_PER.py](Code/DoubleDQN_PER.py)
```
Episode done: cur_frame=556 current_training_step=4192 total_reward=10.0
Episode done: cur_frame=570 current_training_step=4320 total_reward=14.0
Episode done: cur_frame=605 current_training_step=4608 total_reward=35.0
Episode done: cur_frame=805 current_training_step=6208 total_reward=200.0
Episode done: cur_frame=911 current_training_step=7040 total_reward=106.0
Episode done: cur_frame=1051 current_training_step=8160 total_reward=140.0
Episode done: cur_frame=1160 current_training_step=9024 total_reward=109.0
Episode done: cur_frame=1322 current_training_step=10336 total_reward=162.0
Episode done: cur_frame=1404 current_training_step=10976 total_reward=82.0
Episode done: cur_frame=1466 current_training_step=11488 total_reward=62.0
Episode done: cur_frame=1666 current_training_step=13088 total_reward=200.0
```

