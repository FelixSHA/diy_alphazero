import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import datetime

class DQN(nn.Module):
      def __init__(self, input_shape, action_size):
            super(DQN, self).__init__()
            # Defining three linear layers
            self.fc1 = nn.Linear(input_shape, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, action_size)

      def forward(self, state):
            # Forward propagation using ReLU as activation function
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)


class DQNAgent:
      def __init__(self, model, optimizer, game, args):
            self.model = model
            self.optimizer = optimizer
            self.game = game
            self.args = args
            # Initializing memory for storing experiences
            self.memory = []
            # Setting initial exploration rate
            self.epsilon = self.args['epsilon']

      def get_action(self, state):
            # Get valid moves
            valid_moves = self.game.get_valid_moves(state)

            # Reshaping the state and converting it to tensor
            state = torch.FloatTensor(state.reshape(-1, self.game.row_count*self.game.column_count)).unsqueeze(0)

            # Exploration vs Exploitation
            if np.random.rand() <= self.epsilon:
                  # Take random valid action
                  valid_indices = np.where(valid_moves == 1)[0]
                  return random.choice(valid_indices)
            else:
                  # Use the model for action
                  with torch.no_grad():
                        action_values = self.model(state)
                        # Filter out invalid moves
                        action_values[0][valid_moves == 0] = float('-inf')
                        return action_values.argmax().item()

      def train(self):
            # Ensure we have enough experiences in memory to start training
            if len(self.memory) < self.args['batch_size']:
                  return
            # Randomly sampling from the memory for a batch of experiences
            batch = random.sample(self.memory, self.args['batch_size'])

            states, actions, rewards, dones = zip(*batch)
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.append(states[1:], self.game.get_initial_state()))
            dones = torch.BoolTensor(dones)

            # Q-learning updates
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.model(next_states).max(1)[0].detach()
            target_q_values = rewards + self.args['gamma'] * next_q_values * (1 - dones)

            # Compute loss and perform backpropagation
            loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Reduce exploration rate
            if self.epsilon > self.args['epsilon_min']:
                  self.epsilon *= self.args['epsilon_decay']

      def remember(self, state, action, value, done):
            # Store experiences in memory
            self.memory.append((state, action, value, done))

      def learn(self):
            # Initialize state
            state = self.game.get_initial_state()
            player = 1

            for iteration in range(self.args['num_episodes']):
                  state = self.game.get_initial_state()
                  done = False
                  while not done:
                        neutral_state = self.game.change_perspective(state, player)
                        # Get action based on state
                        action = self.get_action(state)
                        # Get next state based on action
                        state = self.game.get_next_state(state, action, player)

                        print(f"Player {player} played action {action} and the state is now:"                        )
                        print(state)
                        print()

                        # Get value and check if game is done
                        value, done = self.game.get_value_and_terminated(state, action)
                        print(value, done)
                        # Check win condition and assign value
                        if value == 1:
                              print(f"Player {player} won!")
                              print()
                              break
                                          
                        # store experience
                        self.remember(state, action, value, done)
                        self.train()
                        player = self.game.get_opponent(player)

            # Save model and optimizer every 'save_every' episodes
            if (iteration+1) % self.args['save_every'] == 0:
                torch.save(self.model.state_dict(), f"model_{iteration}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt")
                torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt")
