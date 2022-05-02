from stablemodels import StableDynamicsModel
from rlmodels import *
import gym
import torch
import numpy as np

env = gym.make('CartPole-v1')

SEED = 1234
env.seed(SEED)


INPUT_DIM = env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = env.action_space.n

actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)
policy = ActorCritic(actor, critic)
policy.apply(init_weights)
ppo_args = PPOArgs()
agent = Agent(policy, None, ppo_args)
agent.load_param("rlmodels/param/ppo_policy.pkl")

# Teting the loaded model
print("| Tesing the loaded rl agent ............ |")
test_rewards = []
for episode in range(1, 10+1):
    test_reward = evaluate(env, agent)
    test_rewards.append(test_reward)
    mean_test_rewards = np.mean(test_rewards[-5:])

    if episode % 10 == 0:
        print(f'| Episode: {episode:3} | Mean Test Rewards: {mean_test_rewards:5.1f} |')

# Define system model
model = StableDynamicsModel((INPUT_DIM,),                 # input shape
                            control_size=1,       # action size
                            alpha=0.9,            # lyapunov constant
                            layer_sizes=[64, 64], # NN layer sizes for lyapunov
                            lr=3e-4,              # learning rate for dynamics model
                            lyapunov_lr=3e-4,     # learning rate for lyapunov function
                            lyapunov_eps=1e-3)    # penalty for equilibrium away from 0
state = env.reset()
state = torch.FloatTensor(state).unsqueeze(0)

# RL agent outputs action
action = agent.take_action(state, training=False)

# Predicted next state
prediction = model(state, torch.tensor(action).reshape(-1,1))

# Actually next state
state, reward, done, _ = env.step(action)
print(prediction, state, reward, done)

