# Report
## Hyper Parameters

num_episodes = 1000\
pretrain_length = 25000\
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
update_type = 'soft'

Network Architectures = (400, 300) # dense layers
