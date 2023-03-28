from stablemodels import StableDynamicsModel
from rlmodels import *
import gym
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import ray
from copy import deepcopy
import subprocess
import logging
import argparse
import time

parser = argparse.ArgumentParser(description='train rl model.')
parser.add_argument('--env', type=int, dest="env", help='start point', default=5)
parser.add_argument('--errweight', type=float, dest="err_weight", help='err_weight', default=5)
parser.add_argument('--seed', type=int, dest="seed", help='random seed', default=123)
parser.add_argument('--version', type=str, dest="version", help='version', default="test_cp")
parser.add_argument('--base', type=str, dest="base", help='base', default="4e-5_ver_3")
parser.add_argument('--plot', action="store_true", dest="if_plot", help='if plot')
args = parser.parse_args()

VERSION = args.version
BASE = args.base
ERR_WEIGHT = args.err_weight
subprocess.run(["mkdir", "-p", f"figs_{VERSION}"])
subprocess.run(["mkdir", "-p", "param"])
subprocess.run(["mkdir", "-p", "logs"])
subprocess.run(["mkdir", "-p", "results"])

# Define logger
logging.basicConfig(
    filename=f"./logs/log_{VERSION}.txt",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment setups
ENV_LIST = ['CartPole-v1', 'MountainCarContinuous-v0', 'Hopper-v4', 'HumanoidStandup-v4', 'Acrobot-v1', 'Pendulum-v1']
ENV_TYPE_LIST = [0, 1, 1, 1, 0, 1]
ROLLOUT_LEN_LIST = [500, 10000, 1000, 1000, 500, 200]
LEARNING_RATE_LIST = [0.001, 0.001, 0.003, 0.003, 0.001, 0.001]
CONTROL_SIZE_LIST = [1, 1, 3, 17, 1, 1]
CONTROL_SCALE_LIST = [1, 1, 1, 1, 1, 2]
STOPPED_TYPE = [True, False, False, False, True, False]
ENV = ENV_LIST[args.env]
IS_CONTINUOUS_ENV = ENV_TYPE_LIST[args.env]
ROLLOUT_LEN = ROLLOUT_LEN_LIST[args.env]
CONTROL_SIZE = CONTROL_SIZE_LIST[args.env]
SAMPLE_EARLY_STOPPED_TRACE_ONLY = STOPPED_TYPE[args.env]
CONTROL_SCALE = CONTROL_SCALE_LIST[args.env]

# Define environment
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device("cpu")
env = gym.make(ENV)

# Define hyperparameters
INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 128

# Load RL model
if not IS_CONTINUOUS_ENV:
    OUTPUT_DIM = env.action_space.n
    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)
    policy = ActorCritic(actor, critic).to(cpu_device)
    policy.apply(init_weights)

else:
    OUTPUT_DIM = env.action_space.shape[0]
    policy = ActorCriticCont(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, CONTROL_SCALE).to(cpu_device)
    policy.apply(init_weights)

PLOT_ONLY = args.if_plot
PRETRAIN = False
NUM_WORKER = os.cpu_count()
NUM_ITER = 800
EPOCH = 50
BATCH_SIZE = 256

ppo_args = PPOArgs(agent_path=f"./rlmodels/param/ppo_policy_{ENV[:4]}_{BASE}.pkl", cont_action=IS_CONTINUOUS_ENV,
                   rollout_len=ROLLOUT_LEN, noise_sigma=0)
rl_optimizer = optim.Adam(policy.parameters(), lr=2e-6)
agent = Agent(policy, rl_optimizer, ppo_args, cpu_device)

# Define system model
model = StableDynamicsModel((INPUT_DIM,),  # input shape
                            control_size=CONTROL_SIZE,  # action size
                            device=device,
                            alpha=0.9,  # lyapunov constant
                            layer_sizes=[64, 64],  # NN layer sizes for lyapunov
                            lr=3e-4,  # learning rate for dynamics model
                            lyapunov_lr=3e-4,  # learning rate for lyapunov function
                            lyapunov_eps=1e-3)  # penalty for equilibrium away from 0
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


@ray.remote
class Worker(object):
    def __init__(self, env_name, agent, device, path="rlmodels/param/ppo_policy.pkl"):
        self.env = gym.make(env_name)
        self.agent = agent
        self.device = device
        self.agent.policy.to(device)
        self.agent.device = device
        # self.agent.load_param(path, device)

    def update_param(self, new_policy):
        self.agent.policy.load_state_dict(new_policy.state_dict())

    def rollout(self, max_step=100, epoch=100, rand=False):

        # t1 = time.time()
        self.agent.policy.to(self.device)
        self.agent.device = self.device

        state_batch = []
        action_batch = []
        next_state_batch = []

        avg_r = 0

        for _ in range(epoch):
            state = self.env.reset()[0]
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            done = False
            cnt = 0
            episode_reward = 0

            state_batch_cur = []
            action_batch_cur = []
            next_state_batch_cur = []

            while not done:
                # RL agent outputs action
                state_batch_cur.append(state)
                action = self.agent.take_action(state, training=False)
                action_batch_cur.append(torch.tensor(action).reshape(-1, CONTROL_SIZE).to(self.device))

                state, reward, done, _, _ = self.env.step(action)
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_state_batch_cur.append(state)

                cnt += 1
                episode_reward += reward

                if cnt >= max_step:
                    break

            if SAMPLE_EARLY_STOPPED_TRACE_ONLY:
                if cnt < max_step:
                    state_batch += state_batch_cur
                    action_batch += action_batch_cur
                    next_state_batch += next_state_batch_cur

            else:
                state_batch += state_batch_cur
                action_batch += action_batch_cur
                next_state_batch += next_state_batch_cur

            avg_r += episode_reward

        return {
            "state": state_batch,
            "action": action_batch,
            "next_state": next_state_batch,
            "reward": episode_reward
        }


if not PLOT_ONLY:
    ray.init()
    avg_errors = []
    avg_rewards = []
    avg_reward_errors = []

    if PRETRAIN:
        RL_PATH = "param/rlmodel_new.pkl"
        with open(f"./figs_{VERSION}/results.json", "r") as json_file:
            data = json.load(json_file)
        avg_errors = data["error"]
        avg_rewards = data["reward"]
        start_ep = len(avg_errors)
        model = torch.load("param/sysmodel.pkl", map_location=device)

    else:
        RL_PATH = f"./rlmodels/param/ppo_policy_{ENV[:4]}_{BASE}.pkl"
        start_ep = 0

    agent.load_param(RL_PATH, device)
    Workers = [Worker.remote(ENV, deepcopy(agent), cpu_device, RL_PATH) for _ in range(NUM_WORKER)]
    agent.policy.to(device)

    high_reward = 0
    for iter in range(start_ep, start_ep + NUM_ITER):
        errors = []
        rewards = []
        reward_errors = []
        reward_history = []

        if iter % 2 == 0:
            if ERR_WEIGHT == 0:
                continue
            # Define system model
            model = StableDynamicsModel((INPUT_DIM,),  # input shape
                                        control_size=CONTROL_SIZE,  # action size
                                        device=device,
                                        alpha=0.9,  # lyapunov constant
                                        layer_sizes=[64, 64],  # NN layer sizes for lyapunov
                                        lr=3e-4,  # learning rate for dynamics model
                                        lyapunov_lr=3e-4,  # learning rate for lyapunov function
                                        lyapunov_eps=1e-3)  # penalty for equilibrium away from 0
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            # t2 = time.time()
            ret = ray.get(
                [worker.rollout.remote(max_step=ROLLOUT_LEN, epoch=EPOCH, rand=int(iter == 0)) for worker in Workers])
            # print("rollout:", time.time() - t2)
            state_batch = []
            action_batch = []
            next_state_batch = []
            for batch in ret:
                state_batch += batch["state"]
                action_batch += batch["action"]
                next_state_batch += batch["next_state"]

            state_batch = torch.cat(state_batch).to(device)
            action_batch = torch.cat(action_batch).to(device)
            next_state_batch = torch.cat(next_state_batch).to(device)

            for ep in range(1):
                error = 0
                # state_batch = []
                # action_batch = []
                # next_state_batch = []
                #
                # t2 = time.time()
                # ret = ray.get([worker.rollout.remote(max_step=5000, rand=int(iter == 0)) for worker in Workers])
                # print("rollout:", time.time() - t2)
                # for batch in ret:
                #     state_batch += batch["state"]
                #     action_batch += batch["action"]
                #     next_state_batch += batch["next_state"]

                # state_batch = torch.cat(state_batch).to(device)
                # action_batch = torch.cat(action_batch).to(device)
                # next_state_batch = torch.cat(next_state_batch).to(device)

                for i in range(state_batch.size(0) // BATCH_SIZE):
                    # Predicted next state
                    prediction = model(state_batch[i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                                       action_batch[i * BATCH_SIZE: (i + 1) * BATCH_SIZE])

                    error = ((prediction - next_state_batch[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]) ** 2).sum(-1)

                    error = error.mean(0)
                    optimizer.zero_grad()
                    error.backward()
                    optimizer.step()

                errors.append(error.item())
                if ep % 100 == 0:
                    logger.info(f"Iter: {iter // 2}, epoch: {ep}, error of dynamic model: {error.item()}")
                torch.save(model, f"./param/sysmodel_{VERSION}.pkl")

        else:
            agent.policy.to(device)
            agent.device = device
            agent.policy.train()

            for ep in range(500):
                error = 0
                n_iter = 0
                state_batch = []
                action_batch = []
                next_state_batch = []
                reward_batch = []
                episode_reward = 0

                state = env.reset()[0]
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                done = False
                cnt = 0

                while not done:
                    # RL agent outputs action
                    state_batch.append(state)
                    action = agent.take_action(state)
                    action_batch.append(torch.tensor(action).reshape(-1, CONTROL_SIZE).to(device))

                    state, reward, done, _, _ = env.step(action)
                    state = torch.FloatTensor(state).unsqueeze(0).to(device)
                    next_state_batch.append(state)
                    reward_batch.append(reward)

                    cnt += 1
                    episode_reward += reward
                    if cnt == ROLLOUT_LEN:
                        break

                state_batch = torch.cat(state_batch)
                action_batch = torch.cat(action_batch)
                next_state_batch = torch.cat(next_state_batch)

                # Predicted next state
                prediction = model(state_batch, action_batch)
                error = ((prediction - next_state_batch) ** 2).sum(-1).detach()
                # error = torch.clamp(error, -100, 100)

                error = error.tolist()
                error_mean = np.mean(error)
                for i in range(len(reward_batch)):
                    if i == 0:
                        agent.update_reward(.1 * reward_batch[i] - ERR_WEIGHT * (error[i] - 0))
                    else:
                        agent.update_reward(.1 * reward_batch[i] - ERR_WEIGHT * (error[i] - error[i - 1]))
                    # agent.update_reward(1 * reward_batch[i] - ERR_WEIGHT * error[i])

                agent.calculate_return_and_adv()
                policy_loss, value_loss = agent.update_policy()
                # print(policy_loss, value_loss)
                rewards.append(episode_reward)
                # reward_errors.append(np.sum(error))
                reward_errors.append(error[-1])
                if ep % 100 == 0:
                    logger.info(
                        f"Iter: {iter // 2}, epoch: {ep}, reward of rl model: {np.mean(rewards[-100:])}, with error: {error_mean}")
            if np.mean(rewards[-100:]) > high_reward:
                agent.save_param(f"./param/rlmodel_new_{VERSION}.pkl")
                high_reward = np.mean(rewards)

            ret = ray.get([worker.update_param.remote(agent.policy.to(cpu_device)) for worker in Workers])

        plt.figure()
        plt.xlabel("Epoch")

        if iter % 2 == 0:
            plt.plot(errors)
            plt.ylabel("Errors of learning dynamic models")
            plt.savefig(f"./figs_{VERSION}/error_{iter // 2}.jpg")
            avg_errors.append(np.mean(errors))
        else:
            plt.plot(rewards)
            plt.plot(np.array(rewards) - np.array(reward_errors))
            plt.ylabel("Rewards of rl models")
            plt.savefig(f"./figs_{VERSION}/reward_{iter // 2}.jpg")
            avg_rewards.append(np.mean(rewards[-100:]))
            avg_reward_errors.append(np.mean(reward_errors))
        plt.close()

    torch.save(model, f"./param/sysmodel_{VERSION}.pkl")
    agent.save_param(f"./param/rlmodel_new_{VERSION}.pkl")

    with open(f"./figs_{VERSION}/results.json", "w") as json_file:
        json.dump(
            {"error": avg_errors, "reward": avg_rewards, "reward_errors": avg_reward_errors},
            json_file)

else:
    with open(f"./figs_{VERSION}/results.json", "r") as json_file:
        data = json.load(json_file)
    with open(f"./figs_cp_error_0_step_5000_epoch_50_iter_500_dist_20/results.json", "r") as json_file:
        data1 = json.load(json_file)

    avg_errors = data["error"]
    avg_rewards = data["reward"]
    avg_reward_errors = data["reward_errors"]
    avg_rewards0 = data1["reward"]

    plt.figure()
    plt.xlabel("Iteration")
    plt.plot(avg_errors)
    plt.ylabel("Error of deep dynamic model")
    plt.ylim([0, 10])
    plt.savefig(f"./figs_{VERSION}/avg_errors.jpg")

    plt.figure()
    plt.xlabel("Iteration")
    plt.plot(avg_rewards)
    # plt.plot(avg_rewards0)
    plt.ylabel("Reward of RL controller (Max: 100)")
    plt.savefig(f"./figs_{VERSION}/avg_rewards.jpg")

    plt.figure()
    plt.xlabel("Iteration")
    plt.plot(avg_reward_errors)
    plt.ylabel("Optimization objective of RL")
    plt.savefig(f"./figs_{VERSION}/avg_loss.jpg")

    torch.save(model, "./param/sysmodel.pkl")
    torch.save(agent, "./param/rlmodel_new.pkl")
