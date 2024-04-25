from stable_learning_control import lac_pytorch as lac
import torch
import torch.nn as nn
import gymnasium as gym

env_fn = lambda : gym.make("stable_gym:CartPoleCost-v1")

ac_kwargs = dict(hidden_sizes={"actor": [256, 64], "critic": [128, 64]}, activation=nn.ReLU)
logger_kwargs = dict(output_dir='path/to/lac_stable_gym:CartPoleCost-v1_dist_20_ver_test', exp_name='lac_stable_gym:CartPoleCost-v1_dist_train')

lac(
    env_fn=env_fn,
    max_ep_len=250,
    epochs=300,
    steps_per_epoch=2048,
    start_steps=0,
    update_every=100,
    update_after=1000,
    steps_per_update=80,
    noise_sigma=20,
    num_test_episodes=10,
    alpha=2.0,
    alpha3=0.1,
    labda=0.99,
    gamma=0.995,
    polyak=0.995,
    adaptive_temperature=True,
    target_entropy=-3,
    lr_a=1e-4,
    lr_c=3e-4,
    lr_a_final=1e-10,
    lr_c_final=1e-10,
    lr_decay_type="linear",
    lr_decay_ref="epoch",
    seed=50,
    save_freq=10,
    logger_kwargs=logger_kwargs,
    ac_kwargs=ac_kwargs
)