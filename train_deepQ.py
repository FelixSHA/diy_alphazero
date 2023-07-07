import torch
from games import ConnectFour
from DeepQ import DQNAgent, DQN

if __name__ == "__main__":
      game = ConnectFour()
      model = DQN(game.row_count*game.column_count, game.action_size)
      optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
      args = {'epsilon': 1.0, 'epsilon_min': 0.01, 'epsilon_decay': 0.995,
            'gamma': 0.95, 'lr': 0.001, 'batch_size': 32,
            'num_episodes': 1000, 'save_every': 100}
      agent = DQNAgent(model, optimizer, game, args)
      agent.learn()

