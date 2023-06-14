import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred


class ActorCriticCont(nn.Module):
    def __init__(self, obs_size, hid_size, act_size, scale=1):
        super(ActorCriticCont, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hid_size, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(hid_size, act_size),
            nn.Softplus(),
        )
        self.value = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1)
        )
        self.scale = scale

    def forward(self, x):
        base_out = self.base(x)
        return self.scale * self.mu(base_out), self.var(base_out) + 1e-4, self.value(x)