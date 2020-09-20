import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from drlnd.project3_tennis.dist_rl.maddpg_agent import MADDPG_Net
from drlnd.project3_tennis.dist_rl.environment import Environment


class args:
    num_episodes = 1000
    eval = False
    train = not eval
    pretrain = 20000
    nographics = False
    actor_learn_rate = 5e-4
    critic_learn_rate = 3e-4
    batch_size = 128
    buffer_size = int(3e5)
    e = 0.3
    e_decay = 1
    e_min = 0.00
    anneal_max = 0.7
    gamma = 0.99
    rollout = 5
    tau = 1e-4
    num_atoms = 100
    vmin = -0.01
    vmax = 2.0
    layer_sizes = [400, 300]
    print_every = 100
    C = 2500
    quit_threshold = 0.8
    update_type = 'soft'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = 0


def train_maddpg(multi_agent, args, env):
    multi_agent.initialize_memory(args.pretrain, env)
    scores_deque = deque(maxlen=args.print_every)
    scores = []
    highest_avg_score = 0
    
    for episode in range(1, args.num_episodes + 1):
        env.reset()
        obs = env.states
        score = np.zeros(multi_agent.agent_count)
        while True:
            actions = multi_agent.act(obs)
            next_obs, rewards, dones = env.step(actions)
            score += rewards
            multi_agent.store((obs, next_obs, actions, rewards, dones))
            multi_agent.learn()
            obs = next_obs
            if np.any(dones):
                break
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)), end="")
        if np.mean(scores_deque) >= highest_avg_score:
            for i, a in enumerate(multi_agent.agents):
                torch.save(a.actor_local.state_dict(), f'checkpoint_actor_a{i}.pth')
                torch.save(a.critic_local.state_dict(), f'checkpoint_critic_a{i}.pth')
            highest_avg_score = np.mean(scores_deque)
        if episode % args.print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
        multi_agent.new_episode(scores)
        if len(scores) > 250:
            if np.array(scores[-250:]).mean() > args.quit_threshold:
                break
    env.close()
    return scores


def load_qnetwork(multi_agent, env):
    for i, a in enumerate(multi_agent.agents):
        a.actor_local.load_state_dict(torch.load(f'checkpoint_actor_a{i}.pth'))
        a.critic_local.load_state_dict(torch.load(f'checkpoint_critic_a{i}.pth'))

    scores = []
    for episode in range(1, 6):
        env.reset(train=False)
        obs = env.states
        score = np.zeros(multi_agent.agent_count)
        while True:
            actions = multi_agent.act(obs, training=False)
            next_obs, rewards, dones = env.step(actions)
            obs = next_obs
            score += rewards
            if np.any(dones):
                break
        scores.append(np.max(score))
        print("Score: {}".format(np.max(score)))
        multi_agent.new_episode(scores)
    env.close()
    return


if __name__ == "__main__":
    env_filepath = "../Tennis_Windows_x86_64/Tennis.exe"
    env = Environment(env_filepath)
    multi_agent = MADDPG_Net(env, args)

    scores = train_maddpg(multi_agent, args, env)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # load_qnetwork(multi_agent, env)
