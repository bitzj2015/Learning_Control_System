from stablemodels import StableDynamicsModel
from rlmodels import *
import gym
import torch
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json

# Define environment
SEED = 1234
env = gym.make('CartPole-v1')

# Define hyperparameters
INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = env.action_space.n
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load RL model
actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)
policy = ActorCritic(actor, critic).to(device)
policy.apply(init_weights)
ppo_args = PPOArgs()
rl_optimizer = optim.Adam(policy.parameters(), lr=1e-2)
agent = Agent(policy, rl_optimizer, ppo_args, device)
agent.load_param("rlmodels/param/ppo_policy.pkl")

# # Testing the loaded RL model
# print("| Tesing the loaded rl agent ............ |")
# test_rewards = []
# for episode in range(1, 6):
#     test_reward = evaluate(env, agent, device)
#     test_rewards.append(test_reward)
#     mean_test_rewards = np.mean(test_rewards[-5:])

#     if episode % 5 == 0:
#         print(f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:5.1f} |')

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

avg_errors = []
avg_rewards = []

for iter in range(50):
    errors = []
    rewards = []
    for ep in range(500):
        error = 0
        n_iter = 0
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        episode_reward = 0

        for _ in range(20):
            state = env.reset()[0]
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            done = False
            cnt = 0

            while not done:
                # RL agent outputs action
                state_batch.append(state)

                if iter % 2 == 0:
                    action = agent.take_action(state, training=False)
                else:
                    action = agent.take_action(state)
                action_batch.append(torch.tensor(action).reshape(-1,1).to(device))

                state, reward, done, _, _ = env.step(action)
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                next_state_batch.append(state)

                if iter % 2 == 1:
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

        if iter % 2 == 0:
            error = error.mean(0)
            error.backward()
            optimizer.step()
            errors.append(error.item())
            if (ep + 1) % 20 == 0:
                print(f"Iter: {iter // 2}, epoch: {ep}, error of dynamic model: {error.item()}")
            
        else:
            error = error.tolist()
            error_max = np.mean(error)
            for i in range(len(reward_batch)):
                agent.update_reward(reward_batch[i] - error[i] / error_max)

            policy_loss, value_loss = agent.update_policy()
            rewards.append(episode_reward)
            if (ep + 1) % 20 == 0:
                print(f"Iter: {iter // 2}, Epoch: {ep}, reward of rl model: {np.mean(rewards)}, with error: {error_max}")

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
plt.plot(avg_errors)
plt.ylabel("Rewards of rl models")
plt.savefig(f"./figs/avg_rewards.jpg")

torch.save(model, "./param/sysmodel.pkl")
torch.save(agent, "./param/rlmodel_new.pkl")

with open("./figs/results.json", "w") as json_file:
    json.dump({"error": avg_errors, "reward": avg_rewards}, json_file)