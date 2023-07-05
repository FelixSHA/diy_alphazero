import numpy as np

from connectFour import ConnectFour

game = ConnectFour()

player = 1

state = game.get_initial_state()

while True:
      print(state)
    
      if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                  print("action not valid")
                  continue
      elif player == -1:
            action = np.random.choice(np.where(game.get_valid_moves(state) == 1)[0])
            print(f"{player}:{action}")
      else:
            raise Exception("invalid player")

      state = game.get_next_state(state, action, player)
      
      value, is_terminal = game.get_value_and_terminated(state, action)
    
      if is_terminal:
            print(state)
            if value == 1:
                  print(f"player: {player}", "won")
            else:
                  print("draw")
            break
        
      player = game.get_opponent(player)
      print()