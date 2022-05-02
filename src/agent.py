import torch
import torch.nn.functional as F
import torch.distributions as distributions


class PPOArgs(object):
    def __init__(
        self,
        ppo_steps=5,
        ppo_clip=0.2,
        discount_factor=0.99,
        normalize=True,
        agent_path="./param/ppo_policy.pkl"
    ):
        self.ppo_steps = ppo_steps
        self.ppo_clip = ppo_clip
        self.discount_factor = discount_factor
        self.normalize = normalize
        self.agent_path = agent_path


class Agent(object):
    def __init__(self, policy, optimizer, args):
        self.policy = policy
        self.optimizer = optimizer
        self.args = args
        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.values = []
        self.rewards = []
        self.returns = []
        self.episode_reward = 0


    def save_param(self):
        torch.save(self.policy.state_dict(), self.args.agent_path)


    def take_action(self, state, training=True):
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

        return action.item()
    

    def update_reward(self, reward):
        self.rewards.append(reward)
        self.episode_reward += reward


    def calculate_returns(self):
        R = 0
        
        # Calculate accumulative rewards as return
        for r in reversed(self.rewards):
            R = r + R * self.args.discount_factor
            self.returns.insert(0, R)
        self.returns = torch.tensor(self.returns)
        
        # Normalize returns
        if self.args.normalize:
            self.returns = (self.returns - self.returns.mean()) / self.returns.std()


    def calculate_advantages(self):
        self.advantages = self.returns - self.values
        
        # Normalize advantages
        if self.args.normalize:
            self.advantages = (self.advantages - self.advantages.mean()) / self.advantages.std()
            

    def update_policy(self):
        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)    
        self.log_prob_actions = torch.cat(self.log_prob_actions)
        self.values = torch.cat(self.values).squeeze(-1)

        self.calculate_returns()
        self.calculate_advantages()
        
        total_policy_loss = 0 
        total_value_loss = 0
        
        self.advantages = self.advantages.detach()
        self.log_prob_actions = self.log_prob_actions.detach()
        self.actions = self.actions.detach()
        
        for _ in range(self.args.ppo_steps):
                    
            # Get new log prob of actions for all input states
            action_pred, value_pred = self.policy(self.states)
            value_pred = value_pred.squeeze(-1)
            action_prob = F.softmax(action_pred, dim = -1)

            dist = distributions.Categorical(action_prob)
            
            # New log prob using old actions
            new_log_prob_actions = dist.log_prob(self.actions)

            policy_ratio = (new_log_prob_actions - self.log_prob_actions).exp()
                    
            policy_loss_1 = policy_ratio * self.advantages
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - self.args.ppo_clip, max = 1.0 + self.args.ppo_clip) * self.advantages
            
            policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
            
            value_loss = F.smooth_l1_loss(self.returns, value_pred).sum()
        
            self.optimizer.zero_grad()

            policy_loss.backward()
            value_loss.backward()

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