from stablemodels import StableDynamicsModel
from rlmodels import *
import gym
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import ray
import time
import os
import subprocess
import logging

subprocess.run(["mkdir", "-p", "figs"])
subprocess.run(["mkdir", "-p", "param"])
subprocess.run(["mkdir", "-p", "logs"])
subprocess.run(["mkdir", "-p", "results"])

# Define environment
SEED = 1234
env = gym.make('CartPole-v1')

# Define hyperparameters
INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = env.action_space.n
PLOT_ONLY = False
NUM_WORKER = os.cpu_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device("cpu")

# Define logger
logging.basicConfig(
    filename=f"./logs/log_test.txt",
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
rl_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
agent = Agent(policy, rl_optimizer, ppo_args, cpu_device)

# Define system model
model = StableDynamicsModel((INPUT_DIM,),                 # input shape
                            control_size=1,       # action size
                            alpha=0.9,            # lyapunov constant
                            layer_sizes=[64, 64], # NN layer sizes for lyapunov
                            lr=3e-4,              # learning rate for dynamics model
                            lyapunov_lr=3e-4,     # learning rate for lyapunov function
                            lyapunov_eps=1e-3)    # penalty for equilibrium away from 0
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


@ray.remote
class Worker(object):
    def __init__(self, env, agent, device):
        self.env = env
        self.agent = agent
        self.device = device
        self.agent.policy.to(device)
        self.agent.device = device
        self.agent.load_param("rlmodels/param/ppo_policy.pkl", device)

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

ray.init()
Workers = [Worker.remote(env, agent, cpu_device) for _ in range(NUM_WORKER)]

if not PLOT_ONLY:
    avg_errors = []
    avg_rewards = []
    start_t = time.time()

    for iter in range(50):
        errors = []
        rewards = []

        if iter % 2 == 0:
            for ep in range(500):
                error = 0
                state_batch = []
                action_batch = []
                next_state_batch = []

                ret = ray.get([worker.rollout.remote() for worker in Workers])
                for batch in ret:
                    state_batch += batch["state"]
                    action_batch += batch["action"]
                    next_state_batch += batch["next_state"]
                
                state_batch = torch.cat(state_batch).to(device)
                action_batch = torch.cat(action_batch).to(device)
                next_state_batch = torch.cat(next_state_batch).to(device)

                # Predicted next state
                prediction = model(state_batch, action_batch)

                error = ((prediction - next_state_batch) ** 2).sum(-1)

                error = error.mean(0)
                error.backward()
                optimizer.step()
                errors.append(error.item())
                if (ep + 1) % 100 == 0:
                    logger.info(f"[{time.time() - start_t}] Iter: {iter // 2}, epoch: {ep}, error of dynamic model: {error.item()}")
        
        else:
            agent.policy.to(device)
            agent.device = device

            for ep in range(200):
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
                    if cnt == 100:
                        break
                
                state_batch = torch.cat(state_batch)
                action_batch = torch.cat(action_batch)
                next_state_batch = torch.cat(next_state_batch)

                # Predicted next state
                prediction = model(state_batch, action_batch)

                error = ((prediction - next_state_batch) ** 2).sum(-1)

                error = error.tolist()
                error_mean = np.mean(error)
                for i in range(len(reward_batch)):
                    agent.update_reward(reward_batch[i] - error[i] / error_mean)

                policy_loss, value_loss = agent.update_policy()
                rewards.append(episode_reward)
                if (ep + 1) % 100 == 0:
                    logger.info(f"[{time.time() - start_t}], epoch: {ep}, reward of rl model: {np.mean(rewards)}, with error: {error_mean}")

            ret = ray.get([worker.update_param.remote(agent.policy.to(cpu_device)) for worker in Workers])

        plt.figure()
        plt.xlabel("Epoch")

        if iter % 2 == 0:
            plt.plot(errors)
            plt.ylabel("Errors of learning dynamic models")
            plt.savefig(f"./figs/error_{iter // 2}.jpg")
            avg_errors.append(np.mean(errors))
        else:
            plt.plot(rewards)
            plt.ylabel("Rewards of rl models")
            plt.savefig(f"./figs/reward_{iter // 2}.jpg")
            avg_rewards.append(np.mean(rewards))
        plt.close()

    plt.figure()
    plt.xlabel("Iteration")
    plt.plot(avg_errors)
    plt.ylabel("Errors of learning dynamic models")
    plt.savefig(f"./figs/avg_errors.jpg")

    plt.figure()
    plt.xlabel("Iteration")
    plt.plot(avg_rewards)
    plt.ylabel("Rewards of rl models")
    plt.savefig(f"./figs/avg_rewards.jpg")

    torch.save(model, "./param/sysmodel.pkl")
    torch.save(agent, "./param/rlmodel_new.pkl")

    with open("./figs/results.json", "w") as json_file:
        json.dump({"error": avg_errors, "reward": avg_rewards}, json_file)

else:
    with open("./figs/results.json", "r") as json_file:
        data = json.load(json_file)

    avg_errors = data["error"]
    avg_rewards = data["reward"]

    plt.figure()
    plt.xlabel("Iteration")
    plt.plot(avg_errors)
    plt.ylabel("Errors of learning dynamic models")
    plt.savefig(f"./figs/avg_errors.jpg")

    plt.figure()
    plt.xlabel("Iteration")
    plt.plot(avg_rewards)
    plt.ylabel("Rewards of rl models")
    plt.savefig(f"./figs/avg_rewards.jpg")

    torch.save(model, "./param/sysmodel.pkl")
    torch.save(agent, "./param/rlmodel_new.pkl")
