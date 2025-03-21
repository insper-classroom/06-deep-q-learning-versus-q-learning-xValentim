import gymnasium as gym
import numpy as np
from tqdm import tqdm
from collections import deque
import random
import torch
import torch.nn as nn

class AgentBase:
    def __init__(self,
                 env,
                 alpha=0.1,
                 gamma=0.999,
                 epsilon = 1.0,      
                 epsilon_min = 0.01, 
                 epsilon_decay = 0.999):
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.size_r = 10
        self.size_v = 100
        self.n_actions = env.action_space.n 
        self.n_states = (env.observation_space.high - env.observation_space.low)*np.array([self.size_r, self.size_v])
        self.n_states = np.round(self.n_states, 0).astype(int) + 1
        
        self.Q = np.zeros([self.n_states[0], 
                           self.n_states[1], 
                           env.action_space.n])
        
    def transform_state(self, state):
        state_adj = (state - self.env.observation_space.low)*np.array([self.size_r, self.size_v])
        return np.round(state_adj, 0).astype(int)

    def get_action(self, state):
        state_adj = self.transform_state(state)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions) 
        else:
            action = np.argmax(self.Q[state_adj[0], state_adj[1]])      
        return action     

    def learn(self, state, action, reward, next_state):
        pass
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def predict(self, state):
        return np.argmax(self.Q[state[0], state[1]])
    
    def save(self, path):
        np.save(path, self.Q)
        print(f"Q-table salva em {path}")
    
    def load(self, path):
        self.Q = np.load(path)
        print(f"Q-table carregada de {path}")
        
class AgentQLearning(AgentBase):
    def __init__(self,
                 env,
                 alpha=0.1,
                 gamma=0.999,
                 epsilon = 1.0,      
                 epsilon_min = 0.01, 
                 epsilon_decay = 0.999):
        super().__init__(env, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
        
    def learn(self, state, action, reward, next_state):
        state = self.transform_state(state)
        next_state = self.transform_state(next_state)
        self.Q[state[0], state[1], action] = self.Q[state[0], state[1], action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state[0], next_state[1]]) - self.Q[state[0], state[1], action])
        
class AgentSarsa(AgentBase):
    def __init__(self,
                 env,
                 alpha=0.1,
                 gamma=0.999,
                 epsilon = 1.0,      
                 epsilon_min = 0.01, 
                 epsilon_decay = 0.999):
        super().__init__(env, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
        
    def learn(self, state, action, reward, next_state):
        state = self.transform_state(state)
        next_action = self.get_action(next_state)
        next_state = self.transform_state(next_state)
        self.Q[state[0], state[1], action] = self.Q[state[0], state[1], action] + self.alpha * (reward + self.gamma * self.Q[next_state[0], next_state[1], next_action] - self.Q[state[0], state[1], action])


class NeuralNetwork(nn.Module):
    def __init__(self, 
                 fc1_dim, 
                 fc2_dim, 
                 input_dim,
                 output_dim):
        
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.output = nn.Linear(fc2_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        # x = self.softmax(x)
        return x
    
class ReplayBuffer:
    """
    Replay buffer básico para armazenar transições e amostrar mini-lotes para treino.
    """
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample_batch(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def size(self):
        return len(self.buffer)
    
class AgentDQN:
    def __init__(self,
                 action_size,
                 observation_size,
                 fc1_dim=128,
                 fc2_dim=128,
                 gamma=0.999,
                 lr=0.01,
                 epsilon = 1.0,      
                 epsilon_min = 0.01, 
                 epsilon_decay = 0.999,
                 batch_size=32,
                 buffer_size=10000,
                 device='cuda',
                 env=None):
        
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.env = env
        self.size_r = 10
        self.size_v = 100
        
        self.n_states = (env.observation_space.high - env.observation_space.low)*np.array([self.size_r, self.size_v])
        self.n_states = np.round(self.n_states, 0).astype(int) + 1
        
        # Replay buffer
        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(max_size=self.buffer_size)
        self.loss_history = []
        print(self.n_states)
        self.policy = NeuralNetwork(fc1_dim=fc1_dim,
                                    fc2_dim=fc2_dim,
                                    input_dim=len(self.n_states),
                                    output_dim=action_size).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.device = device
        
    def store_transition(self, state, action, reward, next_state, done):
        """
        Guarda a transição no replay buffer.
        """
        self.memory.store_transition(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Estratégia epsilon-greedy:
         - Com probabilidade epsilon, escolhe ação aleatória.
         - Caso contrário, escolhe a ação de maior valor Q(s,a).
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state_t = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.policy(state_t)
            # Pega a ação com maior valor Q
            action = torch.argmax(q_values).item()
            return action
    
    def learn(self):
        
        # Só treina se houver amostras suficientes no buffer
        if self.memory.size() < self.batch_size:
            return
        
        
        states, actions, rewards, next_states, dones = self.memory.sample_batch(self.batch_size)
        
         # Converte para tensores
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device).unsqueeze(-1)  # (batch_size, 1)
        rewards_t = torch.FloatTensor(rewards).to(self.device).unsqueeze(-1) # (batch_size, 1)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device).unsqueeze(-1)     # (batch_size, 1)
        
         # Calcula Q(s, a) atual
        q_values = self.policy(states_t)                    # (batch_size, action_size)
        q_values_actions = q_values.gather(1, actions_t)       # Q(s,a) em cada transição

        # Calcula Q*(s', a') = max_a' Q(s', a')
        with torch.no_grad():
            q_next = self.policy(next_states_t)             # (batch_size, action_size)
            q_next_max, _ = torch.max(q_next, dim=1, keepdim=True)  
            # Se for estado terminal (done=1), então valor = 0
            q_target = rewards_t + self.gamma * (1 - dones_t) * q_next_max

        # Calcula a perda
        loss = self.loss_fn(q_values_actions, q_target)

        # Otimiza
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Armazena a perda para monitoramento
        self.loss_history.append(loss.item())

        # Decai epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay