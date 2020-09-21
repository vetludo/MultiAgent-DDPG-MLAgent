# Report
## Hyper Parameters

num_episodes = 1200\
pretrain_length = 20000\
batch_size = 128\
buffer_size = int(3e5)\
actor_learn_rate = 5e-4\
critic_learn_rate = 3e-4\
gamma = 0.99\
tau = 1e-4\
e = 0.3\
e_decay = 1\
e_min = 0.00\
anneal_max = 0.7\
rollout = 5\
num_atoms = 100\
vmin = 0.0\
vmax = 2.0\
update_every = 2500\
print_every = 100\
update_type = 'hard'

Network Architectures = (400, 300) # dense layers

## Description of the algorithm
### Deep Deterministic Policy Gradient (DDPG)
This project is a multi agent cooperation environment. Therefore, I have used Multi Agent Deep Deterministic Policy Gradient (MADDPG) algorithm. Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy. DDPG is usually being used for environments with continuous action spaces like this project.

![ddpq](/assets/DDPG.svg)

### Multi Agent Deep Deterministic Policy Gradient (MADDPG)
The environment of this project consists of two rackets playing tennis. They are trained to pass the balls to each others. Each racket is trained with its own DDPG network and give the observations and actions from all agents to its critic network but only its own observation to its actor network. Therefore, each agents can act with its own observation without knowing the observation and action of other agents.

![maddpq](/assets/MADDPG-algo.png)

The main concept behind this algorithm is summarized in this illustration:
![maddpq_concept](/assets/MADDPG.png)

## Report

The agent is trained by Multi Agent Deep Deterministic Policy Gradient (MADDPG) for this environment. After episode 1100, the agents has already got an average score +0.5 over 100 consecutive episodes.

Episode 100	Average Score: 0.00\
Episode 200	Average Score: 0.03\
Episode 300	Average Score: 0.08\
Episode 400	Average Score: 0.09\
Episode 500	Average Score: 0.10\
Episode 600	Average Score: 0.10\
Episode 700	Average Score: 0.12\
Episode 800	Average Score: 0.13\
Episode 900	Average Score: 0.26\
Episode 1000	Average Score: 0.32\
Episode 1100	Average Score: 0.79\
Episode 1200	Average Score: 1.86

![report](/assets/report.png)
