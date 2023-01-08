from tracemalloc import start
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

VERSION = "test_cp"
subprocess.run(["mkdir", "-p", f"figs_{VERSION}"])
subprocess.run(["mkdir", "-p", "param"])
subprocess.run(["mkdir", "-p", "logs"])
subprocess.run(["mkdir", "-p", "results"])

# Define environment
SEED = 1234
env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')

# Define hyperparameters
INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = env.action_space.n
PLOT_ONLY = False
PRETRAIN = False
NUM_WORKER = os.cpu_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device("cpu")
NUM_ITER = 1000
EPOCH = 500
BATCH_SIZE = 1024

# Define logger
logging.basicConfig(
    filename=f"./logs/log_{VERSION}.txt",
    filemode='w',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 

# Load RL model
actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)
policy = ActorCritic(actor, critic).to(cpu_device)
policy.apply(init_weights)
ppo_args = PPOArgs()
rl_optimizer = optim.Adam(policy.parameters(), lr=1e-4)
agent = Agent(policy, rl_optimizer, ppo_args, cpu_device)

# Define system model
model = StableDynamicsModel((INPUT_DIM,),                 # input shape
                            control_size=1,       # action size
                            device= device, 
                            alpha=0.9,            # lyapunov constant
                            layer_sizes=[64, 64], # NN layer sizes for lyapunov
                            lr=3e-4,              # learning rate for dynamics model
                            lyapunov_lr=3e-4,     # learning rate for lyapunov function
                            lyapunov_eps=1e-3)    # penalty for equilibrium away from 0
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


@ray.remote
class Worker(object):
    def __init__(self, env, agent, device, path="rlmodels/param/ppo_policy.pkl"):
        self.env = gym.make('CartPole-v1')
        self.agent = agent
        self.device = device
        self.agent.policy.to(device)
        self.agent.device = device
        # self.agent.load_param(path, device)

    def update_param(self, new_policy):
        self.agent.policy.load_state_dict(new_policy.state_dict())

    def rollout(self, max_step=100):
        self.agent.policy.to(self.device)
        self.agent.device = self.device

        state_batch = []
        action_batch = []
        next_state_batch = []
        
        state = self.env.reset()[0]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        done = False
        cnt = 0
        episode_reward = 0

        while not done:
            # RL agent outputs action
            state_batch.append(state)
            action = self.agent.take_action(state, training=False)
            action_batch.append(torch.tensor(action).reshape(-1,1).to(self.device))

            state, reward, done, _, _ = self.env.step(action)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_batch.append(state)
            
            cnt += 1
            episode_reward += reward
            if cnt >= max_step:
                break

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
        RL_PATH = "rlmodels/param/ppo_policy.pkl"
        start_ep = 0
    
    agent.load_param("rlmodels/param/ppo_policy.pkl", device)
    Workers = [Worker.remote(deepcopy(env), deepcopy(agent), cpu_device, RL_PATH) for _ in range(NUM_WORKER)]
    agent.policy.to(device)
    

    for iter in range(start_ep, start_ep + NUM_ITER):
        errors = []
        rewards = []
        reward_errors = []

        if iter % 2 == 0:
            for ep in range(EPOCH):
                error = 0
                state_batch = []
                action_batch = []
                next_state_batch = []

                ret = ray.get([worker.rollout.remote(max_step=500) for worker in Workers])
                for batch in ret:
                    state_batch += batch["state"]
                    action_batch += batch["action"]
                    next_state_batch += batch["next_state"]
                
                state_batch = torch.cat(state_batch).to(device)
                action_batch = torch.cat(action_batch).to(device)
                next_state_batch = torch.cat(next_state_batch).to(device)
                
                for i in range(state_batch.size(0) // BATCH_SIZE):
                    # Predicted next state
                    prediction = model(state_batch[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], action_batch[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])
                    error = ((prediction - next_state_batch[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]) ** 2).sum(-1)

                    error = error.mean(0)
                    optimizer.zero_grad()
                    error.backward()
                    optimizer.step()

                errors.append(error.item())
                if ep % 100 == 0:
                    logger.info(f"Iter: {iter // 2}, epoch: {ep}, error of dynamic model: {error.item()}")
        
        else:
            agent.policy.to(device)
            agent.device = device
            agent.policy.train()

            for ep in range(EPOCH):
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
                    action_batch.append(torch.tensor(action).reshape(-1,1).to(device))
                    
                    state, reward, done, _, _ = env.step(action)
                    state = torch.FloatTensor(state).unsqueeze(0).to(device)
                    next_state_batch.append(state)
                    reward_batch.append(reward)
                    
                    cnt += 1
                    episode_reward += reward
                    if cnt == 500:
                        break
                
                state_batch = torch.cat(state_batch)
                action_batch = torch.cat(action_batch)
                next_state_batch = torch.cat(next_state_batch)

                # Predicted next state
                prediction = model(state_batch, action_batch)
                error = ((prediction - next_state_batch) ** 2).sum(-1).detach()

                error = error.tolist()
                error_mean = np.mean(error)
                for i in range(len(reward_batch)):
                    agent.update_reward(1 * reward_batch[i] - 0.5 * error[i])

                agent.calculate_return_and_adv()
                policy_loss, value_loss = agent.update_policy()
                # print(policy_loss, value_loss)
                rewards.append(episode_reward)
                reward_errors.append(np.sum(error))
                if ep % 100 == 0:
                    logger.info(f"Iter: {iter // 2}, epoch: {ep}, reward of rl model: {np.mean(rewards)}, with error: {error_mean}")

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
            avg_rewards.append(np.mean(rewards))
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

    avg_errors = data["error"]
    avg_rewards = data["reward"]
    avg_reward_errors = data["reward_errors"]

    plt.figure()
    plt.xlabel("Iteration")
    plt.plot(avg_errors)
    plt.ylabel("Error of deep dynamic model")
    plt.ylim([0, 10])
    plt.savefig(f"./figs_{VERSION}/avg_errors.jpg")

    plt.figure()
    plt.xlabel("Iteration")
    plt.plot(avg_rewards)
    plt.ylabel("Reward of RL controller (Max: 100)")
    plt.savefig(f"./figs_{VERSION}/avg_rewards.jpg")

    plt.figure()
    plt.xlabel("Iteration")
    plt.plot(avg_reward_errors)
    plt.ylabel("Optimization objective of RL")
    plt.savefig(f"./figs_{VERSION}/avg_loss.jpg")

    torch.save(model, "./param/sysmodel.pkl")
    torch.save(agent, "./param/rlmodel_new.pkl")
