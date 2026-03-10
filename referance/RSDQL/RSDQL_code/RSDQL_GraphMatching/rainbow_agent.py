#-*- coding: utf-8 -*-

import numpy as np
import random
from collections import deque


LEARNING_RATE = 0.001
GAMMA = 0.9
PRIORITY_ALPHA = 0.6
PRIORITY_BETA = 0.4
NOISY_SIGMA = 0.5
N_STEP = 3


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def append(self, transition, td_error=None):
        priority = 1.0 if td_error is None else (abs(td_error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = self.beta
            
        if len(self.buffer) == 0:
            return None
            
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        samples = [self.buffer[i] for i in indices]
        
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        
        obs = np.array([s[0] for s in samples])
        act = np.array([s[1] for s in samples])
        reward = np.array([s[2] for s in samples])
        next_obs = np.array([s[3] for s in samples])
        done = np.array([s[4] for s in samples])
        
        return obs, act, reward, next_obs, done, indices, weights
        
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)


class NStepReplayMemory:
    def __init__(self, capacity, n_step=3, gamma=0.9):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        
    def append(self, transition):
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) >= self.n_step:
            n_transition = self._get_n_step_transition()
            self.buffer.append(n_transition)
            
    def _get_n_step_transition(self):
        reward = 0
        for i in range(self.n_step):
            r, done = self.n_step_buffer[i][2], self.n_step_buffer[i][4]
            reward += (self.gamma ** i) * r
            if done:
                break
                
        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        next_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]
        
        return (state, action, reward, next_state, done)
        
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


class NoisyLinear:
    def __init__(self, in_features, out_features, sigma=0.5):
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        
        self.weight_mu = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.weight_sigma = np.full((in_features, out_features), sigma / np.sqrt(in_features))
        self.weight = self.weight_mu.copy()
        
        self.bias_mu = np.random.randn(out_features) * np.sqrt(2.0 / in_features)
        self.bias_sigma = np.full((out_features,), sigma / np.sqrt(in_features))
        self.bias = self.bias_mu.copy()
        
        self.reset_noise()
        
    def reset_noise(self):
        epsilon_in = np.random.randn(self.in_features)
        epsilon_out = np.random.randn(self.out_features)
        
        self.weight = self.weight_mu + self.weight_sigma * epsilon_out.reshape(1, -1) * epsilon_in.reshape(-1, 1)
        self.bias = self.bias_mu + self.bias_sigma * epsilon_out
        
    def forward(self, x):
        return np.dot(x, self.weight) + self.bias


class DuelingDQN:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, noisy_sigma=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        
        hidden_dim = 128
        
        self.value_stream = NoisyLinear(state_dim, hidden_dim, noisy_sigma)
        self.advantage_stream = NoisyLinear(state_dim, hidden_dim, noisy_sigma)
        
        self.value_head = NoisyLinear(hidden_dim, 1, noisy_sigma)
        self.advantage_head = NoisyLinear(hidden_dim, action_dim, noisy_sigma)
        
        self.optimizer_value = np.zeros_like(self.value_stream.weight_mu)
        self.optimizer_advantage = np.zeros_like(self.advantage_stream.weight_mu)
        
    def _forward(self, state):
        is_single = len(state.shape) == 1
        if is_single:
            state = state.reshape(1, -1)
            
        x_v = np.maximum(0, self.value_stream.forward(state))
        x_a = np.maximum(0, self.advantage_stream.forward(state))
        
        V = self.value_head.forward(x_v)
        A = self.advantage_head.forward(x_a)
        
        A_mean = A - np.mean(A, axis=1, keepdims=True)
        Q = V + A_mean
        
        if is_single:
            return Q[0]
        return Q
        
    def predict(self, state):
        q_values = self._forward(state)
        return np.argmax(q_values)
        
    def get_q_values(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        return self._forward(state)
        
    def reset_noise(self):
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()
        self.value_head.reset_noise()
        self.advantage_head.reset_noise()
        
    def learn(self, states, actions, targets, weights=None):
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
            
        current_q = self._forward(states)
        if len(current_q.shape) == 1:
            current_q = current_q.reshape(1, -1)
        
        td_errors = np.zeros(len(actions))
        for i, action in enumerate(actions):
            td_errors[i] = targets[i] - current_q[i][action]
            
        if weights is not None:
            td_errors = td_errors * weights
            
        return td_errors


class RainbowDQNAgent:
    def __init__(self, state_dim, action_dim, 
                 learning_rate=0.001,
                 gamma=0.9,
                 priority_alpha=0.6,
                 priority_beta=0.4,
                 noisy_sigma=0.5,
                 n_step=3,
                 memory_size=20000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = learning_rate
        self.noisy_sigma = noisy_sigma
        self.n_step = n_step
        
        self.online_net = DuelingDQN(state_dim, action_dim, learning_rate, noisy_sigma)
        self.target_net = DuelingDQN(state_dim, action_dim, learning_rate, noisy_sigma)
        self._update_target_net()
        
        self.memory = NStepReplayMemory(memory_size, n_step, gamma)
        self.prioritized_memory = PrioritizedReplayMemory(memory_size, priority_alpha, priority_beta)
        
        self.learn_counter = 0
        self.target_update_freq = 500
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def _update_target_net(self):
        self.target_net.value_stream.weight_mu = self.online_net.value_stream.weight_mu.copy()
        self.target_net.value_stream.weight_sigma = self.online_net.value_stream.weight_sigma.copy()
        self.target_net.advantage_stream.weight_mu = self.online_net.advantage_stream.weight_mu.copy()
        self.target_net.advantage_stream.weight_sigma = self.online_net.advantage_stream.weight_sigma.copy()
        self.target_net.value_head.weight_mu = self.online_net.value_head.weight_mu.copy()
        self.target_net.value_head.weight_sigma = self.online_net.value_head.weight_sigma.copy()
        self.target_net.advantage_head.weight_mu = self.online_net.advantage_head.weight_mu.copy()
        self.target_net.advantage_head.weight_sigma = self.online_net.advantage_head.weight_sigma.copy()
        
    def _preprocess_state(self, state):
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
            
            if len(features) < self.state_dim:
                features.extend([0] * (self.state_dim - len(features)))
            elif len(features) > self.state_dim:
                features = features[:self.state_dim]
                
            return np.array(features, dtype=np.float32)
        else:
            if hasattr(state, 'flatten'):
                state = state.flatten()
            if len(state) < self.state_dim:
                state = np.concatenate([state, np.zeros(self.state_dim - len(state))])
            elif len(state) > self.state_dim:
                state = state[:self.state_dim]
            return state.astype(np.float32)
            
    def choose_action(self, state, training=True):
        state = self._preprocess_state(state)
        
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
            
        return self.online_net.predict(state)
        
    def store_transition(self, state, action, reward, next_state, done):
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)
        
        transition = (state, action, reward, next_state, done)
        
        self.memory.append(transition)
        
    def learn(self, batch_size=32, weights=None):
        if len(self.memory) < batch_size:
            return 0, []
            
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        current_q = self.online_net.get_q_values(states)
        if len(current_q.shape) == 1:
            current_q = current_q.reshape(1, -1)
        
        current_q_selected = np.array([current_q[i][actions[i]] for i in range(len(actions))])
        
        next_q_online = self.online_net.get_q_values(next_states)
        if len(next_q_online.shape) == 1:
            next_q_online = next_q_online.reshape(1, -1)
        best_actions = np.argmax(next_q_online, axis=1)
        
        next_q_target = self.target_net.get_q_values(next_states)
        if len(next_q_target.shape) == 1:
            next_q_target = next_q_target.reshape(1, -1)
        next_q = np.array([next_q_target[i][best_actions[i]] for i in range(len(best_actions))])
        
        targets = rewards + (1 - dones) * self.gamma * next_q
        
        td_errors = self.online_net.learn(states, actions, targets, weights)
        
        self.learn_counter += 1
        
        if self.learn_counter % self.target_update_freq == 0:
            self._update_target_net()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.online_net.reset_noise()
        
        return np.mean(np.abs(td_errors)), td_errors
        
    def save_model(self, path):
        np.savez(path,
                 value_weight_mu=self.online_net.value_stream.weight_mu,
                 value_weight_sigma=self.online_net.value_stream.weight_sigma,
                 advantage_weight_mu=self.online_net.advantage_stream.weight_mu,
                 advantage_weight_sigma=self.online_net.advantage_stream.weight_sigma,
                 epsilon=self.epsilon)
                 
    def load_model(self, path):
        try:
            data = np.load(path)
            self.online_net.value_stream.weight_mu = data['value_weight_mu']
            self.online_net.value_stream.weight_sigma = data['value_weight_sigma']
            self.online_net.advantage_stream.weight_mu = data['advantage_weight_mu']
            self.online_net.advantage_stream.weight_sigma = data['advantage_weight_sigma']
            self.epsilon = data['epsilon']
            self._update_target_net()
        except:
            pass


def test_rainbow_agent():
    state_dim = 50
    action_dim = 10
    
    agent = RainbowDQNAgent(state_dim, action_dim)
    
    print("=" * 60)
    print("Rainbow DQN Agent Test")
    print("=" * 60)
    
    print("\n--- Test 1: State Preprocessing ---")
    test_state = {
        'resource_features': np.random.randn(8, 3),
        'task_features': np.random.randn(8, 3),
        'match_score': 0.85
    }
    processed = agent._preprocess_state(test_state)
    print(f"  Original state keys: {list(test_state.keys())}")
    print(f"  Processed shape: {processed.shape}")
    
    print("\n--- Test 2: Action Selection ---")
    for i in range(5):
        action = agent.choose_action(test_state, training=True)
        print(f"  Training action {i}: {action}")
        
    print("\n--- Test 3: Store Transitions ---")
    for i in range(100):
        s1 = test_state
        a = random.randint(0, action_dim - 1)
        r = random.random()
        s2 = test_state.copy()
        s2['match_score'] = random.random()
        done = random.random() > 0.9
        agent.store_transition(s1, a, r, s2, done)
    print(f"  Memory size: {len(agent.memory)}")
    
    print("\n--- Test 4: Learning ---")
    td_error, errors = agent.learn(batch_size=32)
    print(f"  TD Error: {td_error:.4f}")
    print(f"  Epsilon: {agent.epsilon:.4f}")
    print(f"  Learn counter: {agent.learn_counter}")
    
    print("\n--- Test 5: Target Network Update ---")
    print(f"  Target update frequency: {agent.target_update_freq}")
    
    print("\n" + "=" * 60)
    print("Rainbow DQN Agent Test Complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_rainbow_agent()
