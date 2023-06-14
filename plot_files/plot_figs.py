import matplotlib.pyplot as plt
import numpy as np
import json

plt.rc('font', family='Times New Roman')

pen_file_ls = ['./pendulum/big_res.json', # 大干扰
           './pendulum/mid_res.json', # 中干扰
           './pendulum/small_res.json'] # 小干扰
base_pen_file_ls = ['./pendulum/big_baseline_res.json', './pendulum/mid_baseline_res.json', './pendulum/small_baseline_res.json']

cartople_file_ls = ['./cartpole/big_res.json', './cartpole/mid_res.json']
base_cartople_file_ls = ['./cartpole/big_baseline_res.json', './cartpole/mid_baseline_res.json']



def read_data(file_ls, base_file_ls, key_ls):
    data_dict = {}
    baseline_dict = {}

    for file_path, key_term in zip(file_ls, key_ls):
        data = json.load(file_path)['rewards']
        data_dict[key_term] = data

    for base_file_path, key_term in zip(base_file_ls, key_ls):
        base_data = json.load(base_file_path)['rewards']
        baseline_dict[key_term] = base_data

    return data_dict, baseline_dict

def plot_3figs(data_dict, baseline_dict, key_ls, title):
    plt.figure(figsize=(4, 12))
    plt.rc('legend', fontsize=12)
    color_ls = ['b', 'orange']

    for i, key_term in enumerate(key_ls):
        plt.subplot(1, len(key_ls), i+1)
        plt.plot(data_dict[key_term], color=color_ls[0], label='xxx', linewidth=1.5)
        plt.plot(baseline_dict[key_term], color=color_ls[1], label='baseline', linewidth=1.5)
        plt.xlabel('Iteration', fontdict=dict(fontsize=14))
        plt.xlabel('Reward of RL controller (Max:100)', fontdict=dict(fontsize=14))
        plt.legend()
        plt.title('xxxx', fontdict=dict(fontsize=14))
    plt.subplots_adjust(wspace=0.2) # 调整子图之间的间隔
    # plt.savefig('./' + title + '.pdf', bbox_inches='tight', pad_inches=0.05, dpi=600)
    # plt.show()


key_ls = ['1', '2', '3']
pen_data_dict, pen_baseline_dict = read_data(file_ls=pen_file_ls[::-1], base_file_ls=base_pen_file_ls[::-1], key_ls=key_ls)
cartople_data_dict, cartople_baseline_dict = read_data(file_ls=cartople_file_ls[::-1], base_file_ls=base_cartople_file_ls[::-1], key_ls=['1', '2'])

plot_3figs(pen_data_dict, pen_baseline_dict, key_ls=key_ls, )
# plot_3figs(cartople_data_dict, cartople_baseline_dict, key_ls=key_ls, )
plt.show()