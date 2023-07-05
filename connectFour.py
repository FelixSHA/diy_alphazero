import numpy as np

class ConnectFour:
      # Connect Four game environment
      def __init__(self):
            self.row_count = 6
            self.column_count = 7
            self.action_size = self.column_count
            self.in_a_row = 4
      # representaion function
      def __repr__(self):
            return "ConnectFour"
            
      # get initial state           
      def get_initial_state(self):
            return np.zeros((self.row_count, self.column_count))
      
      def get_next_state(self, state, action, player):
            row = np.max(np.where(state[:, action] == 0))
            state[row, action] = player
            return state
      
      def get_valid_moves(self, state):
            return (state[0] == 0).astype(np.uint8)
      
      # function, which checks if the game is over


      # def check_win(self, state, action):
      #       if action == None:
      #             return False
            
      #       row = np.min(np.where(state[:, action] != 0))
      #       column = action
      #       player = state[row][column]

      #       def count(offset_row, offset_column):
      #             for i in range(1, self.in_a_row):
      #                   r = row + offset_row * i
      #                   c = action + offset_column * i
      #             if (
      #                   r < 0 
      #                   or r >= self.row_count
      #                   or c < 0 
      #                   or c >= self.column_count
      #                   or state[r][c] != player
      #             ):
      #                   return i - 1
      #             return self.in_a_row - 1

      #       return (
      #             count(1, 0) >= self.in_a_row - 1 # vertical
      #             or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
      #             or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
      #             or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
      #       )

      def check_win(self, board, action):
            # Get row index
            row = (board[:, action] != 0).sum()

            # Define the players
            players = [1, -1]

            # Check vertical, horizontal, and two diagonal lines
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            
            for player in players:
                  for dx, dy in directions:
                        count = 0
                        x, y = row, action
                        while 0 <= x < board.shape[0] and 0 <= y < board.shape[1] and board[x, y] == player:
                              count += 1
                              x, y = x - dx, y - dy

                        x, y = row + dx, action + dy
                        while 0 <= x < board.shape[0] and 0 <= y < board.shape[1] and board[x, y] == player:
                              count += 1
                              x, y = x + dx, y + dy

                        if count >= 4:  # If count is 4 or more, the player has won
                              return player

            return 0  # No player has won yet
      
      def get_value_and_terminated(self, state, action):
            if self.check_win(state, action):
                  return 1, True
            if np.sum(self.get_valid_moves(state)) == 0:
                  return 0, True
            return 0, False
      
      def get_opponent(self, player):
            return -player
      
      def get_opponent_value(self, value):
            return -value
      
      def change_perspective(self, state, player):
            return state * player
      
      def get_encoded_state(self, state):
            encoded_state = np.stack(
                  (state == -1, state == 0, state == 1)
            ).astype(np.float32)
            
            return encoded_state