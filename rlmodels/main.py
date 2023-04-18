import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import gym
from models import *
from utils import *
from agent import *
import subprocess
import argparse

parser = argparse.ArgumentParser(description='train rl model.')
parser.add_argument('--env', type=int, dest="env", help='start point', default=5)
parser.add_argument('--seed', type=int, dest="seed", help='random seed', default=123)
parser.add_argument('--eval', type=int, dest="eval", help='eval', default=0)
parser.add_argument('--dist', type=float, dest="dist", help='dist', default=0)
parser.add_argument('--weight', type=int, dest="weight", help='weight', default=0)
parser.add_argument('--train-base', type=int, dest="train_base", help='train_basecd', default=0)
parser.add_argument('--version', type=str, dest="version", help='version', default="4e-5")
parser.add_argument('--ver', type=str, dest="ver", help='ver', default="1")
args = parser.parse_args()

subprocess.run(["mkdir", "-p", "logs"])
subprocess.run(["mkdir", "-p", "param"])
subprocess.run(["mkdir", "-p", "results"])

ENV_LIST = ['CartPole-v1', 'MountainCarContinuous-v0', 'Hopper-v4', 'HumanoidStandup-v4', 'Acrobot-v1', 'Pendulum-v1']
ENV_TYPE_LIST = [0, 1, 1, 1, 0, 1]
ROLLOUT_LEN_LIST = [500, 10000, 1000, 1000, 500, 200]
LEARNING_RATE_LIST = [0.001, 0.001, 9.8e-5, 4e-5, 0.001, 5e-5]
CONTROL_SCALE_LIST = [1, 1, 1, 0.4, 1, 2]
REWARD_SCALE_ALPHA_LIST = [0, 0, 0, 0, 0, 8.1]
REWARD_SCALE_BETA_LIST = [1, 1, 1, 1, 1, 8.1]
ENV = ENV_LIST[args.env]
VERSION = args.version
IS_CONTINUOUS_ENV = ENV_TYPE_LIST[args.env]
ROLLOUT_LEN = ROLLOUT_LEN_LIST[args.env]
CONTROL_SCALE = CONTROL_SCALE_LIST[args.env]
REWARD_SCALE_ALPHA = REWARD_SCALE_ALPHA_LIST[args.env]
REWARD_SCALE_BETA = REWARD_SCALE_BETA_LIST[args.env]
train_env = gym.make(ENV)
test_env = gym.make(ENV)
# print(torch.cuda.current_device())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())

WEIGHT = args.weight
DIST = args.dist
VER = args.ver
TRAIN_BASE = args.train_base
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
    OUTPUT_DIM = train_env.action_space.shape[0]
    policy = ActorCriticCont(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, CONTROL_SCALE).to(device)
    policy.apply(init_weights)

LEARNING_RATE = LEARNING_RATE_LIST[args.env]
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

ppo_args = PPOArgs(agent_path=f"./param/ppo_policy_{ENV[:4]}.pkl", cont_action=IS_CONTINUOUS_ENV,
                   rollout_len=ROLLOUT_LEN, noise_sigma=DIST,
                   reward_scaling_alpha=REWARD_SCALE_ALPHA,
                   reward_scaling_beta=REWARD_SCALE_BETA)
# ppo_args = PPOArgs(agent_path=f"/home/asd/PycharmProjects/pythonProject1/Learning_Control_System/param/rlmodel_new_cp_error_5_epoch_100_iter_100.pkl", cont_action=IS_CONTINUOUS_ENV, rollout_len=ROLLOUT_LEN)
agent = Agent(policy, optimizer, ppo_args, device)

MAX_EPISODES = 60000
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25
REWARD_MAX = -10000
PRINT_EVERY = 50
PRINT_MAX = 500

EVAL_ONLY = args.eval
train_rewards = []
test_rewards = []

if not EVAL_ONLY:
    for episode in range(1, MAX_EPISODES + 1):

        policy_loss, value_loss, train_reward = train(train_env, agent, device)

        test_reward = evaluate(test_env, agent, device)

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)

        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

        if episode % PRINT_EVERY == 0:
            print(
                f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')

        if mean_test_rewards >= REWARD_MAX:
            REWARD_MAX = mean_test_rewards
            agent.save_param(name=f"./param/ppo_policy_{ENV[:4]}_{VERSION}.pkl")

    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Test Reward')
    plt.plot(train_rewards, label='Train Reward')
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
        # agent.load_param(name=f"../rlmodels/param/ppo_policy_Hopp_9e-5_ver_1.pkl")
        agent.load_param(
            name=f"../param/rlmodel_new_hop_error_{WEIGHT}_step_1000_epoch_50_iter_400_dist_0_ver_{VER}.pkl")
    else:
        if DIST == 0:
            if WEIGHT == 0:
                # agent.load_param(name=f"../param/rlmodel_new_cp_error_{WEIGHT}_epoch_100_iter_200_ver_3.pkl")
                agent.load_param(
                    name=f"../param/rlmodel_new_pen_error_{WEIGHT}_step_500_epoch_50_iter_400_dist_0_ver_000.pkl")
            else:
                agent.load_param(
                    name=f"../param/rlmodel_new_pen_error_{WEIGHT}_step_500_epoch_50_iter_400_dist_0_ver_{VER}.pkl")
        elif DIST == 1.5:
            agent.load_param(
                name=f"../param/rlmodel_new_pen_error_{WEIGHT}_step_500_epoch_50_iter_400_dist_1_5_ver_{VER}.pkl")
        else:
            agent.load_param(
                name=f"../param/rlmodel_new_pen_error_{WEIGHT}_step_500_epoch_50_iter_400_dist_{int(DIST)}_ver_{VER}.pkl")
    for episode in range(1, PRINT_MAX + 1):
        test_reward = evaluate(test_env, agent, device)
        test_rewards.append(test_reward)
        mean_test_rewards = np.mean(test_rewards)

        if episode % PRINT_MAX == 0:
            print(f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
