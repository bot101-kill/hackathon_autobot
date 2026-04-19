import numpy as np

class ReplayBuffer:

    def __init__(self, size=100000):

        self.size = size

        self.ptr = 0

        self.full = False

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []


    def add(
        self,
        s,
        a,
        r,
        s2,
        d
    ):

        if len(self.states) < self.size:

            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_states.append(s2)
            self.dones.append(d)

        else:

            i = self.ptr

            self.states[i] = s
            self.actions[i] = a
            self.rewards[i] = r
            self.next_states[i] = s2
            self.dones[i] = d

            self.ptr = (
                self.ptr + 1
            ) % self.size


    def sample(self, batch_size):

        idx = np.random.randint(
            0,
            len(self.states),
            size=batch_size
        )

        return (

            np.array(self.states)[idx],
            np.array(self.actions)[idx],
            np.array(self.rewards)[idx],
            np.array(self.next_states)[idx],
            np.array(self.dones)[idx]

        )