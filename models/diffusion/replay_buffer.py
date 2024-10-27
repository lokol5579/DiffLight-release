import numpy as np

class ReplayBuffer:
    """
    edit from ac memory
    """
    def __init__(self, agent_num, mini_batch_size, gamma, reward_norm_factor):
        self.agent_num = agent_num
        self.mini_batch_size = mini_batch_size
        self.normalized = False

        self.in_lanes_num = 0

        self.gamma = gamma
        self.reward_norm_factor = reward_norm_factor

        self.states = None
        self.actions = None
        self.next_states = None
        self.rewards = None
        self.obs_mask = None

    def sample(self, shuffle=True):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.mini_batch_size)
        indices = np.arange(n_states, dtype=np.int64)

        if shuffle:
            np.random.shuffle(indices)
        mini_batches = [indices[i:i + self.mini_batch_size] for i in batch_start]

        return self.states, self.actions, \
                self.next_states, self.rewards, \
                self.obs_mask, mini_batches

    def push(self, states, actions, next_states, rewards, obs_mask):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.obs_mask = obs_mask

    def clear(self):
        self.states = None
        self.actions = None
        self.next_states = None
        self.rewards = None
        self.obs_mask = None
