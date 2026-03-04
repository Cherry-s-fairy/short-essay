import random

import HyperParams as hp
from model import Model
import numpy as np
import tensorflow as tf


class Agent:
    def __init__(self, action_dim, state_dim, container_num):
        self.container_num = container_num
        self.deployed = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eval_net = Model(self.action_dim)
        self.target_net = Model(self.action_dim)
        self.target_net.set_weights(self.eval_net.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    def reset(self):
        self.deployed = [0] * self.container_num

    def take_action(self, state):
        if random.Random().random() < hp.epsilon:
            net_output = tf.squeeze(self.eval_net(tf.expand_dims(state, 0)), axis=0)
            action_index = self.argmax_action(net_output)
            if action_index != -1:
                self.deployed[action_index % self.container_num] = 1
        else:
            non_deployed_indices = np.where(np.array(self.deployed) == 0)[0]
            if len(non_deployed_indices) > 0:
                action_index = np.random.choice(non_deployed_indices)
                self.deployed[action_index % self.container_num] = 1
            else:
                action_index = -1
                return action_index
            hp.epsilon = min(1, hp.epsilon + hp.epsilon_decrement)
        return action_index

    def argmax_action(self, net_output):
        index_and_values = [(index, value) for index, value in enumerate(net_output)]
        sorted_index_and_values = sorted(index_and_values, key=lambda x: x[1])
        sorted_index_and_values_reverse = np.array(sorted_index_and_values)[::-1]
        # choose the biggest one without being deployment
        for index_and_value in sorted_index_and_values_reverse:
            action_index = int(index_and_value[0])
            # without being deployed, choose it
            if self.deployed[action_index % self.container_num] == 0:
                return action_index
        return -1

    def learn(self, bs, ba, br, bs_):
        bs = tf.convert_to_tensor(bs)
        ba = tf.convert_to_tensor(ba)
        br = tf.cast(tf.convert_to_tensor(br), dtype=tf.float32)
        bs_ = tf.convert_to_tensor(bs_)
        with tf.GradientTape() as tape:
            q_vals = self.eval_net(bs)
            indices = tf.expand_dims(ba, axis=1)
            gathered_q_vals = tf.gather(q_vals, indices, batch_dims=1)
            q_vals = tf.squeeze(gathered_q_vals, axis=1)
            # q_vals = tf.gather_nd(self.eval_net(bs), tf.expand_dims(ba, axis=-1))
            next_q_vals = tf.math.reduce_max(self.eval_net(bs_), axis=1)
            td_target = br + hp.gamma * next_q_vals
            td_error = tf.losses.mean_squared_error(q_vals, td_target)
            c_grads = tape.gradient(td_error, self.eval_net.trainable_variables)
            self.optimizer.apply_gradients(zip(c_grads, self.eval_net.trainable_weights))
            return td_error

    def update(self):
        eval_weights = self.eval_net.trainable_weights
        target_weights = self.target_net.trainable_weights
        for eval_weight, target_weight in zip(eval_weights, target_weights):
            target_weight.assign(eval_weight)
