from agents.agent import HideAndSeekAgent
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

class PPO(HideAndSeekAgent):
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, eps_clip=0.2):
        self.actor_critic = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits, _ = self.actor_critic(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, states, actions, log_probs, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        _, values = self.actor_critic(states)
        values = values.squeeze()

        # Compute critic loss
        critic_loss = F.mse_loss(values, returns)

        # Compute actor loss
        logits, _ = self.actor_critic(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs_new = dist.log_prob(actions)
        ratio = torch.exp(log_probs_new - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Total loss
        loss = actor_loss + 0.5 * critic_loss

        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()