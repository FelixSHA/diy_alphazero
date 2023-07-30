import torch
from games import ConnectFour
from agents.DeepQ import DQNAgent, DQN

if __name__ == "__main__":
    # Initialize the game
    game = ConnectFour()
    # Initialize the model
    model = DQN(game.row_count*game.column_count, game.action_size) # action size is 7
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # Define the hyperparameters
    args = {'epsilon': 1.0, 'epsilon_min': 0.01, 'epsilon_decay': 0.995,
            'gamma': 0.95, 'lr': 0.001, 'batch_size': 32,
            'num_episodes': 1000, 'save_every': 100}
    # Initialize the agent
    agent = DQNAgent(model, optimizer, game, args)
    # Start learning process
    agent.learn()
