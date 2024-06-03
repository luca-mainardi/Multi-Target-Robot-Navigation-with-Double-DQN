from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    """
    Replay buffer class that stores the agent's transitions a.k.a "memories".
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        """
        Save a transition
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Randomly sample transitions from memory and transpose the 
        batch from batch-array of Transitions to Transition of batch-arrays.
        """
        # Sample transitions
        transitions = random.sample(self.memory, batch_size)
        # Transpose for correct batching
        batch = Transition(*zip(*transitions))
        return batch
    
    def __len__(self):
        return len(self.memory)