# Multi Agent Deep Deterministic Policy Gradient (MADDPG)
Multi Agent Deep Deterministic Policy Gradient (MADDPG) for a tennis game from Unity ML agent

## Introduction
For this project, I work with a Tennis game environment.

![tennis demo](/assets/tennis_intro.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

## Solving the Environment

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Installation

Install Unity ml-agents.
```
git clone https://github.com/Unity-Technologies/ml-agents.git
git -C ml-agents checkout 0.4.0b
pip install ml-agents/python/
```
Install the project requirements.
```
pip install -r requirements.txt
```
Run tennis.ipynb and follow the instruction.

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

## Future Work

- Evaluates the performance of various deep RL algorithms such as TRPO, CEM, CMA-ES and D4PG, on this multi agent tasks to figure out which are best suited.
- Change the rewards system of the environment so that can try multi agents competiting each others instead of cooperation like this project.

Stay tuned.
