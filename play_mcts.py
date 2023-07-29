# Import necessary modules
from games import ConnectFour
from mcts import MCTS

# Initialize game and parameters
game = ConnectFour()
player = 1
args = {
    'C': 2,
    'num_searches': 100
}

# Initialize MCTS
mcts = MCTS(game, args['C'])

# Get the initial game state
state = game.get_initial_state()

# Play the game
while True:
    print(state)
    
    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue
            
    else:
        # Run MCTS for the AI player
        mcts.run(game.change_perspective(state, player), args['num_searches'])
        action = mcts.get_best_move()
        
    state = game.get_next_state(state, action, player)
    
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = game.get_opponent(player)
