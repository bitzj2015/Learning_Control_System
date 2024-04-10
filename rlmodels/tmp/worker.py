import ray
import gym
import torch
import torch.nn.functional as F
import torch.distributions as distributions


@ray.remote
class LocalAgent(object):
    def __init__(self, rl_agent, device, env_name='CartPole-v1'):
        self.rl_agent = rl_agent
        self.state_batch = []
        self.action_batch = []
        self.reward_batch = []
        self.device = device
        self.env_name = gym.make(env_name)
        self.train_rl = False

    def reset_env(self, train_rl=False):
        self.state_batch = []
        self.action_batch = []
        self.log_prob_action_batch = []
        self.reward_batch = []
        self.train_rl = train_rl
        if self.train_rl:
            self.rl_agent.train()
        else:
            self.rl_agent.eval()

        state = self.env.reset()[0]
        self.state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.state_batch.append(self.state)

    def rollout(self, max_step=100):
        done = False
        num_step = 0
        episode_reward = 0

        while not done:
            # RL agent outputs action
            action_pred = self.rl_agent.actor(self.state)
            action_prob = F.softmax(action_pred, dim = -1)
            dist = distributions.Categorical(action_prob)
            action = dist.sample()

            log_prob_action = dist.log_prob(action)
            self.action_batch.append(action.reshape(-1,1))
            self.log_prob_action_batch.append(log_prob_action)

            state, reward, done, _, _ = self.env.step(action.item())
            self.state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.state_batch.append(self.state)

            self.reward_batch.append(reward)
            
            num_step += 1
            episode_reward += reward

            if num_step == max_step:
                break

        return {
            "state": self.state_batch,
            "action": self.action_batch,
            "num_step": num_step,
            "reward": self.reward_batch
        }

    def load_param(self, name=None):
        if name == None:
            self.rl_agent.load_state_dict(torch.load(self.args.agent_path))
        else:
            self.rl_agent.load_state_dict(torch.load(name))


    def update_param(self, global_model_param):
        with torch.no_grad():
            for name, param in self.rl_agent.named_parameters():
                param.data = global_model_param[name].clone().detach()