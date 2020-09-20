import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
from .model import Actor, Critic


class MADDPG_Net:
    def __init__(self, env, args):
        self.t_step = 0
        self.avg_score = 0
        self.C = args.C
        self._e = args.e
        self.e_min = args.e_min
        self.e_decay = args.e_decay
        self.anneal_max = args.anneal_max
        self.update_type = args.update_type
        self.tau = args.tau
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.agent_count = env.agent_count
        self.agents = [DDPG_Agent(self.state_size, self.action_size, args, self.agent_count) for _ in range(self.agent_count)]
        self.batch_size = args.batch_size
        self.memory = ReplayBuffer(args.device, args.buffer_size, args.gamma, args.rollout, self.agent_count)
        self.memory.init_n_step()
        for agent in self.agents:
            self.update_networks(agent, force_hard=True)

    def act(self, obs, training=True):
        with torch.no_grad():
            actions = np.array([agent.act(o) for agent, o in zip(self.agents, obs)])
        if training:
            actions += self._gauss_noise(actions.shape)
        return np.clip(actions, -1, 1)

    def store(self, experience):
        self.memory.store(experience)

    def learn(self):
        self.t_step += 1
        batch = self.memory.sample(self.batch_size)
        obs, next_obs, actions, rewards, dones = batch
        target_actions = [agent.actor_target(next_obs[i]) for i, agent in enumerate(self.agents)]
        predicted_actions = [agent.actor_local(obs[i]) for i, agent in enumerate(self.agents)]
        target_actions = torch.cat(target_actions, dim=-1)
        predicted_actions = torch.cat(predicted_actions, dim=-1)
        obs = obs.transpose(1, 0).contiguous().view(self.batch_size, -1)
        next_obs = next_obs.transpose(1, 0).contiguous().view(self.batch_size, -1)
        for i, agent in enumerate(self.agents):
            agent.learn(obs, next_obs, actions, target_actions, predicted_actions, rewards[i], dones[i])
            self.update_networks(agent)

    def initialize_memory(self, pretrain_length, env):
        print("Initializing memory.")
        obs = env.states
        while len(self.memory) < pretrain_length:
            actions = np.random.uniform(-1, 1, (self.agent_count, self.action_size))
            next_obs, rewards, dones = env.step(actions)
            self.store((obs, next_obs, actions, rewards, dones))
            obs = next_obs
            if np.any(dones):
                env.reset()
                obs = env.states
                self.memory.init_n_step()
        print("memory initialized.")

    @property
    def e(self):
        ylow = self.e_min
        yhigh = self._e
        xlow = 0
        xhigh = self.anneal_max
        steep_mult = 8
        steepness = steep_mult / (xhigh - xlow)
        offset = (xhigh + xlow) / 2
        midpoint = yhigh - ylow
        x = np.clip(self.avg_score, 0, xhigh)
        x = steepness * (x - offset)
        e = ylow + midpoint / (1 + np.exp(x))
        return e

    def new_episode(self, scores):
        avg_across = np.clip(len(scores), 1, 50)
        self.avg_score = np.array(scores[-avg_across:]).mean()
        self.memory.init_n_step()

    def update_networks(self, agent, force_hard=False):
        if self.update_type == "soft" and not force_hard:
            self._soft_update(agent.actor_local, agent.actor_target)
            self._soft_update(agent.critic_local, agent.critic_target)
        elif self.t_step % self.C == 0 or force_hard:
            self._hard_update(agent.actor_local, agent.actor_target)
            self._hard_update(agent.critic_local, agent.critic_target)

    def _soft_update(self, active, target):
        for t_param, param in zip(target.parameters(), active.parameters()):
            t_param.data.copy_(self.tau * param.data + (1 - self.tau) * t_param.data)

    def _hard_update(self, active, target):
        target.load_state_dict(active.state_dict())

    def _gauss_noise(self, shape):
        n = np.random.normal(0, 1, shape)
        return self.e * n


class DDPG_Agent:
    def __init__(self, state_size, action_size, args, agent_count=1, l2_decay=0.0001):
        self.device = args.device
        self.eval = args.eval
        self.actor_learn_rate = args.actor_learn_rate
        self.critic_learn_rate = args.critic_learn_rate
        self.gamma = args.gamma
        self.rollout = args.rollout
        self.num_atoms = args.num_atoms
        self.vmin = args.vmin
        self.vmax = args.vmax
        self.atoms = torch.linspace(self.vmin, self.vmax, self.num_atoms).to(self.device)
        self.atoms = self.atoms.unsqueeze(0)
        self.actor_local = Actor(state_size, action_size, args.random_seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, args.random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_learn_rate, weight_decay=l2_decay)
        all_states_size = state_size * agent_count
        all_actions_size = action_size * agent_count
        self.critic_local = Critic(all_states_size, all_actions_size, self.num_atoms, args.random_seed).to(self.device)
        self.critic_target = Critic(all_states_size, all_actions_size, self.num_atoms, args.random_seed).to(self.device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=self.critic_learn_rate, weight_decay=l2_decay)

    def act(self, obs, eval=False):
        obs = obs.to(self.device)
        with torch.no_grad():
            actions = self.actor_local(obs).cpu().numpy()
        return actions

    def learn(self, obs, next_obs, actions, target_actions, predicted_actions, rewards, dones):
        log_probs = self.critic_local(obs, actions, log=True)
        target_probs = self.critic_target(next_obs, target_actions).detach()
        target_dist = self._categorical(rewards, target_probs, dones)
        critic_loss = -(target_dist * log_probs).sum(-1).mean()
        probs = self.critic_local(obs, predicted_actions)
        expected_reward = (probs * self.atoms).sum(-1)
        actor_loss = -expected_reward.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    def _categorical(self, rewards, probs, dones):
        vmin = self.vmin
        vmax = self.vmax
        atoms = self.atoms
        num_atoms = self.num_atoms
        gamma = self.gamma
        rollout = self.rollout
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1).type(torch.float)
        delta_z = (vmax - vmin) / (num_atoms - 1)
        projected_atoms = rewards + gamma ** rollout * atoms * (1 - dones)
        projected_atoms.clamp_(vmin, vmax)
        b = (projected_atoms - vmin) / delta_z
        precision = 1
        b = torch.round(b * 10 ** precision) / 10 ** precision
        lower_bound = b.floor()
        upper_bound = b.ceil()
        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs
        projected_probs = torch.tensor(np.zeros(probs.size())).to(self.device)
        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()


class ReplayBuffer:
    def __init__(self, device, buffer_size=100000, gamma=0.99, rollout=5, agent_count=1):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.gamma = gamma
        self.rollout = rollout
        self.agent_count = agent_count

    def store(self, experience):
        if self.rollout > 1:
            self.n_step.append(experience)
            if len(self.n_step) < self.rollout:
                return
            experience = self._n_stack()
        obs, next_obs, actions, rewards, dones = experience
        actions = torch.from_numpy(np.concatenate(actions)).float()
        rewards = torch.from_numpy(rewards).float()
        dones = torch.tensor(dones).float()

        self.buffer.append((obs, next_obs, actions, rewards, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, k=batch_size)
        obs, next_obs, actions, rewards, dones = zip(*batch)
        obs = torch.stack(obs).transpose(1, 0).to(self.device)
        next_obs = torch.stack(next_obs).transpose(1, 0).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).transpose(1, 0).to(self.device)
        dones = torch.stack(dones).transpose(1, 0).to(self.device)
        return (obs, next_obs, actions, rewards, dones)

    def init_n_step(self):
        self.n_step = deque(maxlen=self.rollout)

    def _n_stack(self):
        obs, next_obs, actions, rewards, dones = zip(*self.n_step)
        summed_rewards = rewards[0]
        for i in range(1, self.rollout):
            summed_rewards += self.gamma ** i * rewards[i]
            if np.any(dones[i]):
                break
        obs = obs[0]
        nstep_obs = next_obs[i]
        actions = actions[0]
        dones = dones[i]
        return (obs, nstep_obs, actions, summed_rewards, dones)

    def __len__(self):
        return len(self.buffer)
