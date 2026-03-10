#-*- coding: utf-8 -*-

import os
import sys
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import Env
from dataSet.data import Data
from rainbow_agent import RainbowDQNAgent


LEARNING_RATE = 0.001
GAMMA = 0.9
MEMORY_SIZE = 10000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 32
LEARN_FREQ = 8
MAX_EPISODE = 200


class SimpleDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.q_table = {}
        
        self.memory = []
        self.memory_size = MEMORY_SIZE
        self.memory_ptr = 0
        
    def _get_state_key(self, state):
        if isinstance(state, dict):
            features = []
            resource_features = state.get('resource_features', [])
            if hasattr(resource_features, 'flatten'):
                resource_features = resource_features.flatten()
            features.extend(resource_features.tolist() if hasattr(resource_features, 'tolist') else list(resource_features))
            
            task_features = state.get('task_features', [])
            if hasattr(task_features, 'flatten'):
                task_features = task_features.flatten()
            features.extend(task_features.tolist() if hasattr(task_features, 'tolist') else list(task_features))
            
            match_score = state.get('match_score', 0)
            features.append(match_score)
            
            key = tuple([round(x, 2) for x in features[:10]])
            return key
        return tuple(state[:10])
        
    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        key = self._get_state_key(state)
        
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_dim)
            
        return np.argmax(self.q_table[key])
        
    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) < self.memory_size:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.memory_ptr] = (state, action, reward, next_state, done)
            self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
            
    def learn(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0
            
        batch = random.sample(self.memory, batch_size)
        
        total_loss = 0
        
        for state, action, reward, next_state, done in batch:
            key = self._get_state_key(state)
            next_key = self._get_state_key(next_state)
            
            if key not in self.q_table:
                self.q_table[key] = np.zeros(self.action_dim)
            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(self.action_dim)
            
            current_q = self.q_table[key][action]
            target_q = reward + (1 - done) * self.gamma * np.max(self.q_table[next_key])
            
            self.q_table[key][action] += LEARNING_RATE * (target_q - current_q)
            
            total_loss += abs(target_q - current_q)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return total_loss / batch_size
    
    def save_model(self, path):
        np.savez(path, q_table=self.q_table, epsilon=self.epsilon)
        
    def load_model(self, path):
        try:
            data = np.load(path, allow_pickle=True).item()
            self.q_table = data.get('q_table', {})
            self.epsilon = data.get('epsilon', 1.0)
        except:
            pass


def run_episode(env, agent, is_rainbow=True):
    obs, valid, message = env.reset()
    
    total_reward = 0
    step = 0
    loss = 0
    
    while True:
        action = agent.choose_action(obs, training=True)
        next_obs, reward, done, info = env.step(action)
        
        agent.store_transition(obs, action, reward, next_obs, done)
        
        total_reward += reward
        step += 1
        
        if len(agent.memory) > MEMORY_WARMUP_SIZE and step % LEARN_FREQ == 0:
            if is_rainbow:
                l, _ = agent.learn(batch_size=BATCH_SIZE)
            else:
                l = agent.learn(batch_size=BATCH_SIZE)
            loss = l
            
        obs = next_obs
        if done:
            break
            
    return total_reward, step, loss, info


def evaluate(env, agent, is_rainbow=True):
    obs, valid, message = env.reset()
    
    total_reward = 0
    step = 0
    
    while True:
        action = agent.choose_action(obs, training=False)
        next_obs, reward, done, info = env.step(action)
        
        total_reward += reward
        step += 1
        obs = next_obs
        
        if done:
            break
            
    return total_reward, step, info


def run_comparison():
    print("=" * 70)
    print("        DQN vs Rainbow DQN Comparison Experiment")
    print("=" * 70)
    
    data = Data('./dataSet/data.xml')
    node_count = len(data.uav_nodes)
    service_count = len(data.service_nodes)
    print(f"\nLoaded data: {node_count} nodes, {service_count} services")
    
    env = Env('./dataSet/data.xml')
    
    obs_dim = node_count * 3 + service_count * 3 + 1
    act_dim = 10
    
    print(f"\nState dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")
    print(f"Max episodes: {MAX_EPISODE}")
    
    print("\n" + "=" * 70)
    print("Training Simple DQN Agent...")
    print("=" * 70)
    
    dqn_agent = SimpleDQNAgent(obs_dim, act_dim)
    
    dqn_rewards = []
    dqn_costs = []
    dqn_scores = []
    
    episode = 0
    while len(dqn_agent.memory) < MEMORY_WARMUP_SIZE:
        run_episode(env, dqn_agent, is_rainbow=False)
        print(f"  DQN Warmup: {len(dqn_agent.memory)}/{MEMORY_WARMUP_SIZE}", end='\r')
    
    print("\n  DQN Training started...")
    
    while episode < MAX_EPISODE:
        total_reward, step, loss, info = run_episode(env, dqn_agent, is_rainbow=False)
        
        episode += 1
        
        dqn_rewards.append(total_reward)
        dqn_costs.append(info.get('cost', 0))
        dqn_scores.append(info.get('match_score', 0))
        
        if episode % 20 == 0:
            eval_reward, _, eval_info = evaluate(env, dqn_agent, is_rainbow=False)
            print(f"  DQN Episode {episode:3d} | Reward: {total_reward:8.2f} | "
                  f"Score: {info.get('match_score', 0):.4f} | "
                  f"Epsilon: {dqn_agent.epsilon:.3f} | Eval: {eval_reward:8.2f}")
    
    print("\n" + "=" * 70)
    print("Training Rainbow DQN Agent...")
    print("=" * 70)
    
    rainbow_agent = RainbowDQNAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        priority_alpha=0.6,
        priority_beta=0.4,
        noisy_sigma=0.5,
        n_step=3,
        memory_size=MEMORY_SIZE
    )
    
    rainbow_rewards = []
    rainbow_costs = []
    rainbow_scores = []
    
    episode = 0
    while len(rainbow_agent.memory) < MEMORY_WARMUP_SIZE:
        run_episode(env, rainbow_agent, is_rainbow=True)
        print(f"  Rainbow Warmup: {len(rainbow_agent.memory)}/{MEMORY_WARMUP_SIZE}", end='\r')
    
    print("\n  Rainbow DQN Training started...")
    
    while episode < MAX_EPISODE:
        total_reward, step, loss, info = run_episode(env, rainbow_agent, is_rainbow=True)
        
        episode += 1
        
        rainbow_rewards.append(total_reward)
        rainbow_costs.append(info.get('cost', 0))
        rainbow_scores.append(info.get('match_score', 0))
        
        if episode % 20 == 0:
            eval_reward, _, eval_info = evaluate(env, rainbow_agent, is_rainbow=True)
            print(f"  Rainbow Episode {episode:3d} | Reward: {total_reward:8.2f} | "
                  f"Score: {info.get('match_score', 0):.4f} | "
                  f"Epsilon: {rainbow_agent.epsilon:.3f} | Eval: {eval_reward:8.2f}")
    
    print("\n" + "=" * 70)
    print("                    Comparison Results")
    print("=" * 70)
    
    print("\n┌─────────────────┬──────────────┬──────────────────┐")
    print("│     Metric      │    Simple DQN │    Rainbow DQN   │")
    print("├─────────────────┼──────────────┼──────────────────┤")
    
    dqn_final_reward = np.mean(dqn_rewards[-20:])
    rainbow_final_reward = np.mean(rainbow_rewards[-20:])
    print(f"│ Avg Reward      │ {dqn_final_reward:12.2f} │ {rainbow_final_reward:16.2f} │")
    
    dqn_final_score = np.mean(dqn_scores[-20:])
    rainbow_final_score = np.mean(rainbow_scores[-20:])
    print(f"│ Avg Match Score │ {dqn_final_score:12.4f} │ {rainbow_final_score:16.4f} │")
    
    dqn_final_cost = np.mean(dqn_costs[-20:])
    rainbow_final_cost = np.mean(rainbow_costs[-20:])
    print(f"│ Avg Cost        │ {dqn_final_cost:12.2f} │ {rainbow_final_cost:16.2f} │")
    
    dqn_max_score = np.max(dqn_scores)
    rainbow_max_score = np.max(rainbow_scores)
    print(f"│ Max Match Score │ {dqn_max_score:12.4f} │ {rainbow_max_score:16.4f} │")
    
    improvement = (rainbow_final_score - dqn_final_score) / max(dqn_final_score, 0.001) * 100
    print(f"│ Score Improve   │              │ {improvement:+14.2f}% │")
    
    print("└─────────────────┴──────────────┴──────────────────┘")
    
    print("\n[Summary]")
    if rainbow_final_score > dqn_final_score:
        print(f"  ✓ Rainbow DQN outperforms Simple DQN by {improvement:.2f}%")
    else:
        print(f"  ✗ Simple DQN performs better by {-improvement:.2f}%")
    
    print("\n  Simple DQN: Uses Q-table with epsilon-greedy exploration")
    print("  Rainbow DQN: Dueling + Double DQN + Prioritized Replay + NoisyNet + N-Step")
    
    return {
        'dqn': {'rewards': dqn_rewards, 'scores': dqn_scores, 'costs': dqn_costs},
        'rainbow': {'rewards': rainbow_rewards, 'scores': rainbow_scores, 'costs': rainbow_costs}
    }


if __name__ == '__main__':
    results = run_comparison()
