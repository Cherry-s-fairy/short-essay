#-*- coding: utf-8 -*-

import os
import sys
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import Env
from dataSet.data import Data


LEARN_FREQ = 8
MEMORY_SIZE = 20000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.9
MAX_EPISODE = 1000


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        
    def append(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs = np.array([t[0] for t in batch])
        act = np.array([t[1] for t in batch])
        reward = np.array([t[2] for t in batch])
        next_obs = np.array([t[3] for t in batch])
        done = np.array([t[4] for t in batch])
        
        return obs, act, reward, next_obs, done
        
    def __len__(self):
        return len(self.buffer)


class SimpleAgent:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.weights = np.random.randn(obs_dim, act_dim) * 0.1
        
    def predict(self, obs):
        if isinstance(obs, dict):
            features = np.concatenate([
                obs['resource_features'].flatten(),
                obs['task_features'].flatten(),
                [obs['match_score']]
            ])
        else:
            features = obs
            
        if len(features) != self.obs_dim:
            features = np.random.randn(self.obs_dim)
            
        Q = np.dot(features, self.weights)
        return np.argmax(Q)
        
    def learn(self, obs, act, reward, next_obs, done):
        if isinstance(obs, dict):
            features = np.concatenate([
                obs['resource_features'].flatten(),
                obs['task_features'].flatten(),
                [obs['match_score']]
            ])
        else:
            features = obs
            
        if len(features) != self.obs_dim:
            return 0.0
            
        Q = np.dot(features, self.weights)
        target = reward + GAMMA * np.max(Q) * (1 - done)
        
        error = target - Q[act]
        self.weights[:, act] += LEARNING_RATE * error * features
        
        return np.abs(error)


def run_episode(env, agent, rpm):
    obs, valid, message = env.reset()
    
    total_reward = 0
    step = 0
    loss_list = []
    
    while True:
        action = agent.predict(obs)
        
        next_obs, reward, done, info = env.step(action)
        
        rpm.append((obs, action, reward, next_obs, done))
        
        total_reward += reward
        step += 1
        
        if len(rpm) > MEMORY_WARMUP_SIZE and step % LEARN_FREQ == 0:
            batch_obs, batch_act, batch_reward, batch_next_obs, batch_done = rpm.sample(BATCH_SIZE)
            
            loss = 0
            for i in range(BATCH_SIZE):
                l = agent.learn(batch_obs[i], batch_act[i], batch_reward[i], 
                              batch_next_obs[i], batch_done[i])
                loss += l
            loss_list.append(loss / BATCH_SIZE)
            
        obs = next_obs
        
        if done:
            break
            
    avg_loss = np.mean(loss_list) if loss_list else 0
    return total_reward, step, avg_loss, info


def evaluate(env, agent):
    obs, valid, message = env.reset()
    
    total_reward = 0
    step = 0
    
    while True:
        action = agent.predict(obs)
        next_obs, reward, done, info = env.step(action)
        
        total_reward += reward
        step += 1
        obs = next_obs
        
        if done:
            break
            
    return total_reward, step, info


def plot_results(episode_rewards, episode_costs, episode_scores):
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 3, 2)
        plt.plot(episode_costs)
        plt.title('Episode Costs')
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        
        plt.subplot(1, 3, 3)
        plt.plot(episode_scores)
        plt.title('Match Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.close()
        
        print("Training results saved to training_results.png")
    except ImportError:
        print("Matplotlib not available, skipping plot")


def main():
    print("=" * 50)
    print("RSDQL Graph Matching Training")
    print("=" * 50)
    
    data = Data('./dataSet/data.xml')
    print(f"Loaded data: {data.NodeNumber} nodes, {data.ServiceNumber} services")
    
    env = Env('./dataSet/data.xml')
    print(f"Resource Graph: {env.resource_graph}")
    print(f"Task Graph: {env.task_graph}")
    
    obs_dim = data.NodeNumber * 3 + data.ServiceNumber * 3 + 1
    act_dim = 10
    
    agent = SimpleAgent(obs_dim, act_dim)
    rpm = ReplayMemory(MEMORY_SIZE)
    
    print("\n" + "=" * 50)
    print("Starting Training")
    print("=" * 50)
    
    episode_rewards = []
    episode_costs = []
    episode_scores = []
    
    episode = 0
    
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)
        print(f"Warmup: {len(rpm)}/{MEMORY_WARMUP_SIZE}")
        
    print("\nTraining started...")
    
    while episode < MAX_EPISODE:
        total_reward, step, avg_loss, info = run_episode(env, agent, rpm)
        
        episode += 1
        
        episode_rewards.append(total_reward)
        episode_costs.append(info.get('cost', 0))
        episode_scores.append(info.get('match_score', 0))
        
        if episode % 50 == 0:
            eval_reward, _, eval_info = evaluate(env, agent)
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:8.2f} | "
                  f"Cost: {info.get('cost', 0):8.2f} | "
                  f"Score: {info.get('match_score', 0):.4f} | "
                  f"Eval Reward: {eval_reward:8.2f}")
            
            with open("reward.txt", "a") as f:
                f.write(f"{episode:05d},{total_reward:.3f}\n")
            with open("cost.txt", "a") as f:
                f.write(f"{episode},{info.get('cost', 0):.6f}\n")
                
    print("\n" + "=" * 50)
    print("Training Completed")
    print("=" * 50)
    
    plot_results(episode_rewards, episode_costs, episode_scores)
    
    final_eval_reward, _, final_info = evaluate(env, agent)
    print(f"\nFinal Evaluation:")
    print(f"  Reward: {final_eval_reward:.2f}")
    print(f"  Cost: {final_info.get('cost', 0):.2f}")
    print(f"  Match Score: {final_info.get('match_score', 0):.4f}")
    print(f"  Success Rate: {final_info.get('metrics', {}).get('success_rate', 0):.2%}")
    print(f"  Avg Latency: {final_info.get('metrics', {}).get('avg_latency', 0):.2f}ms")
    print(f"  Reschedule Count: {final_info.get('metrics', {}).get('reschedule_count', 0)}")
    
    return final_info


if __name__ == '__main__':
    main()
