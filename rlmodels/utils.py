import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


def train(env, agent, device, step_flag=True):
    # accu_steps = 0
    done = False
    episode_reward = 0
    state = env.reset()[0]

    num_steps = 0
    while not done:

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        #append state here, not after we get the next state from env.step()
        action = agent.take_action(state)

        state, reward, done, _, _ = env.step(action)
        num_steps += 1
        agent.update_reward(1-reward/20)
        episode_reward += reward

        if num_steps >= agent.args.rollout_len:
            break

    # accu_steps += num_steps
    agent.calculate_return_and_adv()
    policy_loss, value_loss = agent.update_policy()

    return policy_loss, value_loss, episode_reward, num_steps


def evaluate(env, agent, device, step_flag=True):
    done = False
    episode_reward = 0
    state = env.reset()[0]
    num_steps = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = agent.take_action(state, training=False)
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        num_steps += 1
        if num_steps >= agent.args.rollout_len:
            break

    return episode_reward, num_steps
