import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import datetime

class DQN(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game      
        self.args = args
        self.memory = []
        self.epsilon = self.args['epsilon']


    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(-1, self.game.row_count*self.game.column_count)).unsqueeze(0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.game.action_size)
        else:
            with torch.no_grad():
                return self.model(state).argmax().item()

    def train(self):
        if len(self.memory) < self.args['batch_size']:
            return
        batch = random.sample(self.memory, self.args['batch_size'])

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q_values = rewards + self.args['gamma'] * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.args['epsilon_min']:
            self.epsilon *= self.args['epsilon_decay']

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        state = self.game.get_initial_state()
        for iteration in range(self.args['num_episodes']):
            done = False
            while not done:
                action = self.get_action(state)
                next_state = self.game.get_next_state(state, action, 1)
                reward, done = self.game.get_value_and_terminated(state, action)
                if self.game.check_win(next_state, action):
                    reward = 1
                elif not np.any(next_state==0):
                    reward = 0
                else:
                    reward = -0.01
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.train()

            if (iteration+1) % self.args['save_every'] == 0:
                torch.save(self.model.state_dict(), f"model_{iteration}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt")
                torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt")
