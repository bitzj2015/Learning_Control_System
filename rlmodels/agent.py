import torch
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


class PPOArgs(object):
    def __init__(
        self,
        rollout_len=500,
        ppo_steps=5,
        ppo_clip=0.2,
        discount_factor=0.99,
        normalize=False,
        agent_path="./param/ppo_policy.pkl",
        cont_action=False,
        noise_sigma=0.2,
        reward_scaling_alpha=8.1,
        reward_scaling_beta=8.1
    ):
        self.ppo_steps = ppo_steps
        self.ppo_clip = ppo_clip
        self.discount_factor = discount_factor
        self.normalize = normalize
        self.agent_path = agent_path
        self.cont_action = cont_action
        self.rollout_len = rollout_len
        self.noise_sigma = noise_sigma
        self.alpha = reward_scaling_alpha
        self.beta = reward_scaling_beta


class Agent(object):
    def __init__(self, policy, optimizer, args, device):
        self.policy = policy
        self.optimizer = optimizer
        self.args = args
        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.values = []
        self.rewards = []
        self.returns = []
        self.episode_reward = 0.2
        self.device = device


    def load_param(self, name=None, device=torch.device('cpu')):
        if name == None:
            self.policy.load_state_dict(torch.load(self.args.agent_path, map_location=device))
        else:
            self.policy.load_state_dict(torch.load(name, map_location=device))


    def save_param(self, name=None):
        if name == None:
            torch.save(self.policy.state_dict(), self.args.agent_path)
        else:
            torch.save(self.policy.state_dict(), name)


    def take_action(self, state, training=True):
        # import pdb
        # pdb.set_trace()
        if not self.args.cont_action:
            if training:
                self.states.append(state)
                self.policy.train()
                action_pred, value_pred = self.policy(state)
                action_prob = F.softmax(action_pred, dim = -1)
                dist = distributions.Categorical(action_prob)
                action = dist.sample()

                log_prob_action = dist.log_prob(action)
                self.actions.append(action)
                self.log_prob_actions.append(log_prob_action)
                self.values.append(value_pred)

            else:
                self.policy.eval()
                with torch.no_grad():
                    action_pred, _ = self.policy(state)
                    action_prob = F.softmax(action_pred, dim = -1)
                    dist = distributions.Categorical(action_prob)
                    action = dist.sample()

            if np.random.random() < self.args.noise_sigma:
                return 1 - action.item()
            
            return action.item()

        else:
            if training:
                self.states.append(state)
                self.policy.train()
                action_mu, action_std, value_pred = self.policy(state)
                dist = distributions.Normal(action_mu, action_std)
                action = dist.sample()
                action = torch.clamp(action, min=-20, max=20)

                log_prob_action = dist.log_prob(action)
                self.actions.append(action)
                self.log_prob_actions.append(log_prob_action.sum(-1).reshape(-1,1))
                self.values.append(value_pred)
                action = action[0].cpu().numpy().astype(np.float64)
            else:
                self.policy.eval()
                with torch.no_grad():
                    action_mu, action_std, _ = self.policy(state)
                    dist = distributions.Normal(action_mu, action_std)
                    action = dist.sample()
                    action = torch.clamp(action, min=-20, max=20)
                    action = action[0].cpu().numpy().astype(np.float64)

            # noise = torch.normal(0, self.args.noise_sigma, size=action.size()).to(self.device)
            # action = torch.clamp(action + noise, min=-2, max=2)
            
            
            # return action.tolist()[0]
            return action


    def update_reward(self, reward):
        self.rewards.append((reward + self.args.alpha)/ self.args.beta)
        self.episode_reward += reward


    def calculate_returns(self):
        R = 0
        
        cur_returns = []
        # Calculate accumulative rewards as return
        for r in reversed(self.rewards):
            R = r + R * self.args.discount_factor
            cur_returns.insert(0, R)

        self.returns += cur_returns


    def calculate_advantages(self):
        self.advantages = self.returns - self.values
        
        # Normalize advantages
        if self.args.normalize:
            self.advantages = (self.advantages - self.advantages.mean()) / self.advantages.std()
            self.returns = (self.returns - self.returns.mean()) / self.returns.std()
    

    def calculate_return_and_adv(self):
        self.calculate_returns()
        self.rewards = []


    def update_policy(self):
        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)    
        self.log_prob_actions = torch.cat(self.log_prob_actions)
        self.values = torch.cat(self.values).squeeze(-1)
        self.returns = torch.tensor(self.returns).to(self.device)

        self.calculate_advantages()
        
        total_policy_loss = 0 
        total_value_loss = 0
        
        self.advantages = self.advantages.detach()
        self.log_prob_actions = self.log_prob_actions.detach()
        self.actions = self.actions.detach()
        
        for _ in range(self.args.ppo_steps):
            
            if not self.args.cont_action:
                # Get new log prob of actions for all input states
                action_pred, value_pred = self.policy(self.states)
                value_pred = value_pred.squeeze(-1)
                action_prob = F.softmax(action_pred, dim = -1)
                dist = distributions.Categorical(action_prob)
                new_log_prob_actions = dist.log_prob(self.actions)

            else:
                action_mu, action_std, value_pred = self.policy(self.states)
                value_pred = value_pred.squeeze(-1)
                dist = distributions.Normal(action_mu, action_std)
                # New log prob using old actions
                new_log_prob_actions = dist.log_prob(self.actions).sum(-1).reshape(-1,1)

            policy_ratio = (new_log_prob_actions - self.log_prob_actions).exp()
                    
            policy_loss_1 = policy_ratio * self.advantages
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - self.args.ppo_clip, max = 1.0 + self.args.ppo_clip) * self.advantages
            
            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.smooth_l1_loss(self.returns, value_pred).mean()
        
            self.optimizer.zero_grad()

            policy_loss.backward(retain_graph=True)
            value_loss.backward(retain_graph=True)

            self.optimizer.step()
        
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        self.clear()
        return total_policy_loss / self.args.ppo_steps, total_value_loss / self.args.ppo_steps


    def clear(self):
        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.values = []
        self.rewards = []
        self.returns = []
        self.episode_reward = 0