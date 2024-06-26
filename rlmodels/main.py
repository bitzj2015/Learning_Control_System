import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from stable_gym.envs.classic_control.cartpole_cost.cartpole_cost import CartPoleCost

import stable_learning_control
from stable_learning_control.algos.pytorch.lac.lac import LAC, lac
from models import *
from utils import *
from agent import *
import subprocess
import argparse

parser = argparse.ArgumentParser(description='train rl model.')
parser.add_argument('--env', type=int, dest="env", help='start point', default=0)
parser.add_argument('--seed', type=int, dest="seed", help='random seed', default=0)
parser.add_argument('--eval', type=int, dest="eval", help='eval', default=0)
parser.add_argument('--dist-arg', type=str, dest="dist_arg", help='dist_arg', default="0")
parser.add_argument('--dist', type=float, dest="dist", help='dist', default=0)
parser.add_argument('--weight', type=str, dest="weight", help='weight', default=0)
parser.add_argument('--train-base', type=int, dest="train_base", help='train_basecd', default=1)
parser.add_argument('--test-base', type=int, dest="test_base", help='test_basecd', default=1)
parser.add_argument('--version', type=str, dest="version", help='version', default="4e-5")
parser.add_argument('--ver', type=str, dest="ver", help='ver', default="1")
args = parser.parse_args()

subprocess.run(["mkdir", "-p", "logs"])
subprocess.run(["mkdir", "-p", "param"])
subprocess.run(["mkdir", "-p", "results"])

ENV_LIST = ['stable_gym:CartPoleCost-v1', 'MountainCarContinuous-v0', 'Hopper-v4', 'HumanoidStandup-v4', 'Acrobot-v1',
            'Pendulum-v1']
ENV_TYPE_LIST = [1, 1, 1, 1, 0, 1]
ROLLOUT_LEN_LIST = [250, 10000, 1000, 1000, 500, 200]
LEARNING_RATE_LIST = [1e-5, 0.001, 1e-5, 4e-5, 0.001, 5e-5]
CONTROL_SCALE_LIST = [20, 1, 1, 0.4, 1, 2]
REWARD_SCALE_ALPHA_LIST = [0, 0, 0, 0, 0, 8.1]
REWARD_SCALE_BETA_LIST = [250, 1, 10, 1, 1, 8.1]
ENV = ENV_LIST[args.env]
VERSION = args.version
IS_CONTINUOUS_ENV = ENV_TYPE_LIST[args.env]
ROLLOUT_LEN = ROLLOUT_LEN_LIST[args.env]
CONTROL_SCALE = CONTROL_SCALE_LIST[args.env]
REWARD_SCALE_ALPHA = REWARD_SCALE_ALPHA_LIST[args.env]
REWARD_SCALE_BETA = REWARD_SCALE_BETA_LIST[args.env]
train_env = gym.make(ENV)
test_env = gym.make(ENV)


WEIGHT = args.weight
DIST = args.dist
DIST_ARG = args.dist_arg
VER = args.ver
TRAIN_BASE = args.train_base
TEST_BASE = args.test_base
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 128

if not IS_CONTINUOUS_ENV:
    OUTPUT_DIM = train_env.action_space.n
    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1).to(device)
    policy = ActorCritic(actor, critic).to(device)
    policy.apply(init_weights)

else:
    # print(train_env.action_space)
    OUTPUT_DIM = train_env.action_space.shape[0]
    policy = ActorCriticCont(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, CONTROL_SCALE).to(device)
    policy.apply(init_weights)

LEARNING_RATE = LEARNING_RATE_LIST[args.env]
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

ppo_args = PPOArgs(agent_path=f"./param/ppo_policy_{ENV[:4]}.pkl", cont_action=IS_CONTINUOUS_ENV,
                   rollout_len=ROLLOUT_LEN, noise_sigma=DIST,
                   reward_scaling_alpha=REWARD_SCALE_ALPHA,
                   reward_scaling_beta=REWARD_SCALE_BETA)

# ppo_args = PPOArgs(agent_path=f"/home/asd/PycharmProjects/pythonProject1/Learning_Control_System/param/rlmodel_new_cp_error_5_epoch_100_iter_100.pkl", cont_action=IS_CONTINUOUS_ENV, rollout_len=ROLLOUT_LEN)
agent = Agent(policy, optimizer, ppo_args, device)

MAX_EPISODES = 100000
DISCOUNT_FACTOR = 0.99
N_TRIALS = 50
REWARD_MAX = -10000
PRINT_EVERY = 50
PRINT_MAX = 5000

EVAL_ONLY = args.eval
train_rewards = []
test_rewards = []
train_steps = []
test_steps = []

if not EVAL_ONLY:
    for episode in range(1, MAX_EPISODES + 1):

        flag = True
        if episode % PRINT_EVERY == 0:
            flag = False

        policy_loss, value_loss, train_reward, train_step = train(train_env, agent, device, step_flag=flag)

        test_reward, test_step = evaluate(test_env, agent, device, step_flag=flag)

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        train_steps.append(train_step)
        test_steps.append(test_step)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        mean_train_steps = np.mean(train_steps[-N_TRIALS:])
        mean_test_steps = np.mean(test_steps[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:
            print(
                f'| Episode: {episode:3} | Mean Train Steps: {mean_train_steps:5.1f} | Mean Test steps: {mean_test_steps:5.1f} |')

        if mean_test_rewards >= REWARD_MAX:
            REWARD_MAX = mean_test_steps
            agent.save_param(name=f"./param/ppo_policy_{ENV[:4]}_{VERSION}.pkl")

    plt.figure(figsize=(12, 8))
    plt.plot(test_steps, label='Test Reward')
    plt.plot(train_steps, label='Train Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)
    # plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f"./results/ppo_{ENV[:4]}_{VERSION}.jpg")

else:
    # agent.load_param(name=f"./param/ppo_policy_{ENV[:4]}.pkl")
    print("weight:", WEIGHT)
    if TRAIN_BASE == 1:
        if TEST_BASE == 1:
            agent.load_param(name=f"../param/rlmodel_new_cp_error_5_epoch_10_iter_300_dist_20_ver_1.pkl")
        else:
            agent.load_param(
                name=f"../param/rlmodel_new_hop_error_{WEIGHT}_step_1000_epoch_50_iter_400_dist_{DIST_ARG}_ver_{VER}.pkl")
    elif TRAIN_BASE == 2:
        agent.load_param(
            name=f"../param/rlmodel_new_cp_error_{WEIGHT}_step_500_epoch_100_iter_300_dist_{DIST_ARG}_ver_{VER}.pkl")
    else:
        if DIST == 0:
            if WEIGHT == 0:
                agent.load_param(
                    name=f"../param/rlmodel_new_pen_error_{WEIGHT}_step_500_epoch_50_iter_400_dist_0_ver_000.pkl")
            else:
                agent.load_param(
                    name=f"../param/rlmodel_new_pen_error_{WEIGHT}_step_500_epoch_50_iter_400_dist_0_ver_{VER}.pkl")
        elif DIST == 1.5:
            agent.load_param(
                name=f"../param/rlmodel_new_pen_error_{WEIGHT}_step_500_epoch_50_iter_400_dist_{DIST_ARG}_ver_{VER}.pkl")
        else:
            agent.load_param(
                name=f"../param/rlmodel_new_pen_error_{WEIGHT}_step_500_epoch_50_iter_400_dist_{DIST_ARG}_ver_{VER}.pkl")
    for episode in range(1, PRINT_MAX + 1):
        test_reward, test_step = evaluate(test_env, agent, device)
        # print(test_step)
        test_rewards.append(test_reward)
        test_steps.append(test_step)
        mean_test_rewards = np.mean(test_rewards)
        mean_test_steps = np.mean(test_steps)

        if episode % PRINT_MAX == 0:
            print(f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_steps} |')
