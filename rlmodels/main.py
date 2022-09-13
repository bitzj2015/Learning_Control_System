import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym
from models import *
from utils import *
from agent import *

train_env = gym.make('CartPole-v1')
test_env = gym.make('CartPole-v1')

SEED = 1234
# train_env.seed(SEED)
# test_env.seed(SEED+1)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = train_env.action_space.n


actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1).to(device)
policy = ActorCritic(actor, critic).to(device)
policy.apply(init_weights)


LEARNING_RATE = 0.01
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)


ppo_args = PPOArgs()
agent = Agent(policy, optimizer, ppo_args, device)


MAX_EPISODES = 500
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25
REWARD_THRESHOLD = 475
PRINT_EVERY = 10

EVAL_ONLY = True
train_rewards = []
test_rewards = []

if not EVAL_ONLY:
    for episode in range(1, MAX_EPISODES+1):
        
        policy_loss, value_loss, train_reward = train(train_env, agent, device)
        
        test_reward = evaluate(test_env, agent, device)
        
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        
        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
        
        if mean_test_rewards >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            agent.save_param()
            break


    plt.figure(figsize=(12,8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("./results/test.jpg")

else:
    agent.load_param()
    for episode in range(1, PRINT_EVERY+1):
        test_reward = evaluate(test_env, agent, device)
        test_rewards.append(test_reward)
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:
            print(f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:5.1f} |')