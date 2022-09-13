import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


def train(env, agent, device):
    done = False
    episode_reward = 0

    state = env.reset()[0]

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        #append state here, not after we get the next state from env.step()
        action = agent.take_action(state)
        state, reward, done, _, _ = env.step(action)

        agent.update_reward(reward)
        
        episode_reward += reward
    
    policy_loss, value_loss = agent.update_policy()

    return policy_loss, value_loss, episode_reward


def evaluate(env, agent, device):
    done = False
    episode_reward = 0

    state = env.reset()[0]

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = agent.take_action(state, training=False)
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        
    return episode_reward