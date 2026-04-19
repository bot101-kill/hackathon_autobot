import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)


    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)

        log_std = self.log_std(x)
        log_std = torch.clamp(
            log_std,
            LOG_STD_MIN,
            LOG_STD_MAX
        )

        return mean, log_std


    def sample(self, state):

        mean, log_std = self.forward(state)

        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)

        x_t = normal.rsample()

        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(
            1 - action.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob



class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.fc1 = nn.Linear(
            state_dim + action_dim,
            256
        )

        self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(256, 1)


    def forward(self, state, action):

        x = torch.cat([state, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)