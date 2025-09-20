import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        h, w, c = state_dim
        self.obs_shape = (c, h, w)

        self.conv1 = nn.Conv2d(c, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        self.cnn = nn.Sequential(
            self.conv1, nn.ReLU(),
            self.conv2, nn.ReLU(),
            self.conv3, nn.ReLU(),
            self.conv4, nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feat_dim = self.cnn.cpu()(dummy).shape[1]
        
        self.linear = nn.Linear(feat_dim, 512)
        
        self.actor = nn.Sequential(
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(512, 1)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, np.sqrt(2.0))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        if hasattr(self, "linear"):
            nn.init.orthogonal_(self.linear.weight, np.sqrt(2.0))
            nn.init.constant_(self.linear.bias, 0.0)

        policy_head = self.actor[0] if isinstance(self.actor, nn.Sequential) else self.actor
        value_head  = self.critic[0] if isinstance(self.critic, nn.Sequential) else self.critic

        nn.init.orthogonal_(policy_head.weight, 0.01)
        nn.init.constant_(policy_head.bias, 0.0)

        nn.init.orthogonal_(value_head.weight, 1.0)
        nn.init.constant_(value_head.bias, 0.0)

    def forward(self):
        raise NotImplementedError

    def _to_cnn_input(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)

        c_expected, h_expected, w_expected = self.obs_shape

        if x.dim() == 3:
            if x.shape[0] == c_expected and x.shape[1] == h_expected and x.shape[2] == w_expected:
                x = x.unsqueeze(0)
            elif x.shape[0] == h_expected and x.shape[1] == w_expected and x.shape[2] == c_expected:
                x = x.permute(2, 0, 1).unsqueeze(0)
            else:
                raise RuntimeError(f"Unexpected 3D shape: {tuple(x.shape)}")
        elif x.dim() == 4:
            if x.shape[1] == c_expected and x.shape[2] == h_expected and x.shape[3] == w_expected:
                pass
            elif x.shape[3] == c_expected and x.shape[1] == h_expected and x.shape[2] == w_expected:
                x = x.permute(0, 3, 1, 2)
            elif x.shape == (c_expected, h_expected, w_expected, 1):
                x = x.squeeze(-1).unsqueeze(0)
            else:
                raise RuntimeError(f"Unexpected 4D shape: {tuple(x.shape)}")
        else:
            raise RuntimeError(f"Expected 3D or 4D tensor, got {x.dim()}D")

        return x / 255.0
    
    def act(self, state):
        x = self._to_cnn_input(state)
        features = self.cnn(x)
        features = F.relu(self.linear(features))
        
        action_probs = self.actor(features)
        dist = Categorical(action_probs)
            
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(features)
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        x = self._to_cnn_input(state)
        features = self.cnn(x)
        features = F.relu(self.linear(features))
        
        action_probs = self.actor(features)
        dist = Categorical(action_probs)
            
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, n_epochs, eps_clip, gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, batch_size=64):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = batch_size
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.cnn.parameters(),    'lr': lr_actor},
            {'params': self.policy.linear.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_array = state
            else:
                state_array = np.array(state)
            state = torch.from_numpy(state_array).float().to(device)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.buffer.is_terminals, dtype=torch.float32, device=device)

        next_values = torch.cat([old_state_values[1:], torch.tensor([0.0], device=device)])
        next_values = next_values * (1.0 - dones)
        deltas = rewards + self.gamma * next_values - old_state_values

        advantages = torch.zeros_like(rewards, device=device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + old_state_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = advantages.shape[0]
        for _ in range(self.K_epochs):
            indices = torch.randperm(N, device=device)
            for start in range(0, N, self.minibatch_size):
                end = min(start + self.minibatch_size, N)
                mb_idx = indices[start:end]

                mb_states = old_states[mb_idx]
                mb_actions = old_actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx].detach()

                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                state_values = torch.squeeze(state_values)

                ratios = torch.exp(logprobs - mb_old_logprobs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.MseLoss(state_values, mb_returns).mean()
                entropy_loss = dist_entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))





