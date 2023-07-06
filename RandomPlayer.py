import numpy as np
import time

class RandomPlayer:
      def __init__(self, game, args):
            self.game = game
            self.args = args

      def selfPlay(self):
            np.random.seed(args['random_state'])
            memory = []
            player = 1
            state = self.game.get_initial_state()

            while True:
                  neutral_state = self.game.change_perspective(state, player)

                  memory.append((neutral_state, None, player))
                  action = np.random.choice(self.game.action_size, )

                  state = self.game.get_next_state(state, action, player)

                  time.sleep(0.5)
                  print(
                        f"Player {player} played action {action} and the state is now:"
                  )
                  print(state)
                  print()

                  value, is_terminal = self.game.get_value_and_terminated(state, action)

                  if is_terminal:
                        returnMemory = []
                        for hist_neutral_state, hist_action_probs, hist_player in memory:
                              hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                              returnMemory.append((
                                    self.game.get_encoded_state(hist_neutral_state),
                                    hist_action_probs,
                                    hist_outcome
                              ))
                        return returnMemory

                  player = self.game.get_opponent(player)

from games import ConnectFour

if __name__ == "__main__":
      game = ConnectFour()
      args = {'random_state': 42}
      player = RandomPlayer(game, args)
      player.selfPlay()