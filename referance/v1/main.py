import os
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from Environment import Environment
from agent import Agent
import HyperParams as hp
from replaybuffer import ReplayBuffer


def plot_cost(costs):
    plt.figure(1)
    plt.clf()
    plt.title('Train')
    plt.xlabel('Episodes(*50)')
    plt.ylabel('cost')
    plt.plot(costs)
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_reward(rewards):
    plt.figure(1)
    plt.clf()
    plt.title('Train')
    plt.xlabel('Episodes(*50)')
    plt.ylabel('reward')
    plt.plot(rewards)
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_loss(losses):
    plt.figure(1)
    plt.clf()
    plt.title('Train')
    plt.xlabel('Episodes(*50)')
    plt.ylabel('loss')
    plt.plot(losses)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    # random_seed = 42
    # random.seed(random_seed)  # set random seed for python
    # np.random.seed(random_seed)  # set random seed for numpy
    # tf.random.set_seed(random_seed)  # set random seed for tensorflow-cpu
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'  # set random seed for tensorflow-gpu
    tf.config.experimental.set_visible_devices(tf.config.experimental.list_physical_devices('GPU')[0], 'GPU')

    RB = ReplayBuffer(hp.replyBuffer_size)
    env = Environment()
    action_dim = env.action_dim
    state_dim = env.state_dim
    container_num = env.container_num
    agent = Agent(action_dim, action_dim, container_num)
    episode_t = 0
    total_step_num = 0
    rewards = []
    reward_tmp = []
    costs = []
    costs_tmp = []
    losses = []
    start_train = False
    while episode_t < hp.max_episode:
        # store all information of each step in an episode
        # [state, action, next_state]
        state = env.reset()
        agent.reset()
        done = False
        episode_t += 1
        episode_s = []
        episode_s_ = []
        episode_a = []
        step_num = 0
        while not done:
            step_num += 1
            action_index = agent.take_action(state)
            next_state, done = env.step(action_index)
            # print("episode:{} \t step:{} \t action = {} -> {}".format(episode_t, step_num, action_index % container_num, int(action_index / container_num)))
            episode_s.append(state)
            episode_s_.append(next_state)
            episode_a.append(action_index)
            total_step_num += 1
            if done:
                # calculate cost and reward after this episode is done
                cost, _, _ = env.cost()
                reward, better = env.reward(episode_t, cost)
                if better:
                    push_num = 10
                else:
                    push_num = 1
                # print(reward)
                costs_tmp.append(cost)
                reward_tmp.append(reward)
                if (len(costs_tmp) == 10):
                    avg_cost = sum(costs_tmp) / len(costs_tmp)
                    avg_reward = sum(reward_tmp) / len(reward_tmp)
                    costs.append(avg_cost)
                    costs_tmp.clear()
                    rewards.append(avg_reward)
                    reward_tmp.clear()
                    # print("episode:{} \t avg_cost:{} \t avg_reward:{}".format(episode_t, avg_cost, avg_reward))
                    # reward may be positive and negative, add them all to the queue
                for i in range(len(episode_a)):
                    for _ in range(push_num):
                        RB.push(episode_s[i], episode_a[i], reward, episode_s_[i])
            if total_step_num > hp.replyBuffer_size:
                bs, ba, br, bs_ = RB.sample(hp.batch_size)
                td_error = agent.learn(bs, ba, br, bs_)
                td_error.numpy()
                losses.append(td_error)
                start_train = True
            if total_step_num % hp.target_update_frequency == 0:
                agent.update()
        if episode_t % hp.plot_frequency == 0:
            plot_loss(losses)
            plot_reward(rewards)
            plot_cost(costs)
