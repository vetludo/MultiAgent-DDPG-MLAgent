# Report
## Hyper Parameters

N_EPISODE = 500\
MAX_T = 1000\
BUFFER_SIZE = int(5e5)  # replay buffer size\
BATCH_SIZE = 128  # minibatch size\
GAMMA = 0.99  # discount factor\
TAU = 1e-3  # for soft update of target parameters\
UPDATE_EVERY = 350\
UPDATE_TYPE = 'hard'\
LR_ACTOR = 5e-4  # learning rate of the actor\
LR_CRITIC = 7.5e-4  # learning rate of the critic\
WEIGHT_DECAY = 0  # L2 weight decay\
E_START = 0.2\
E_MIN = 0.03\
E_DECAY = 0.999\
V_MIN = 0\
V_MAX = 0.3\
NUM_ATOMS = 100

Network Architectures = (400, 300) # dense layers
