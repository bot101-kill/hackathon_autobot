import torch
import torch.nn.functional as F
from .sac_network import Actor, Critic


class SACAgent:

    def __init__(
        self,
        state_dim,
        action_dim
    ):

        self.device = "cpu"

        self.actor = Actor(
            state_dim,
            action_dim
        ).to(self.device)

        self.q1 = Critic(
            state_dim,
            action_dim
        ).to(self.device)

        self.q2 = Critic(
            state_dim,
            action_dim
        ).to(self.device)


        self.q1_target = Critic(
            state_dim,
            action_dim
        ).to(self.device)

        self.q2_target = Critic(
            state_dim,
            action_dim
        ).to(self.device)


        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())


        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=1e-4
        )

        self.q1_opt = torch.optim.Adam(
            self.q1.parameters(),
            lr=1e-4
        )

        self.q2_opt = torch.optim.Adam(
            self.q2.parameters(),
            lr=1e-4
        )


        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.1


    def select_action(self, state):

        state = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        action, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]


    def update(self, buffer, batch_size=64):

        s, a, r, s2, d = buffer.sample(batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)


        with torch.no_grad():

            a2, logp = self.actor.sample(s2)

            q1_t = self.q1_target(s2, a2)
            q2_t = self.q2_target(s2, a2)

            q_t = torch.min(q1_t, q2_t)

            target = r + self.gamma * (1-d) * (
                q_t - self.alpha * logp
            )


        q1_loss = F.mse_loss(
            self.q1(s, a),
            target
        )

        q2_loss = F.mse_loss(
            self.q2(s, a),
            target
        )


        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()


        a_new, logp_new = self.actor.sample(s)

        q1_new = self.q1(s, a_new)
        q2_new = self.q2(s, a_new)

        q_new = torch.min(q1_new, q2_new)

        actor_loss = (
            self.alpha * logp_new - q_new
        ).mean()


        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


        for target, source in zip(
            self.q1_target.parameters(),
            self.q1.parameters()
        ):

            target.data.copy_(

                target.data * (1-self.tau)
                + source.data * self.tau

            )


        for target, source in zip(
            self.q2_target.parameters(),
            self.q2.parameters()
        ):

            target.data.copy_(

                target.data * (1-self.tau)
                + source.data * self.tau

            )