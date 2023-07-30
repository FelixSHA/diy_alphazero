from games import ConnectFour
from agents.RandomPlayer import RandomPlayer

if __name__ == "__main__":
      game = ConnectFour()
      args = {
            'random_state': 42
            }
      player = RandomPlayer(game, args)
      player.selfPlay()