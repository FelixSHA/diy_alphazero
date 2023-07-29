import numpy as np

class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
        
    def __repr__(self):
        return "ConnectFour"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    # def get_next_state(self, state, action, player):
    #     row = np.max(np.where(state[:, action] == 0))
    #     state[row, action] = player
    #     return state

    def get_next_state(self, state, action, player):
        """
        Get the game state after a given action.
        The player drops a piece in the specified column (action).
        """
        # Find the lowest available row in the column
        row = np.max(np.where(state[:, action] == 0), initial=-1)
        if row == -1:
            # If the column is full, return the current state
            return state

        next_state = state.copy()
        next_state[row, action] = player
        return next_state


    
    def get_valid_moves(self, state):
        if len(state.shape) == 3:
            return (state[:, 0] == 0).astype(np.uint8)
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row_count
                    or c < 0 
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
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
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state
    
    def get_single_array_state(self, state):
        """
        Return the state as a single 2D array. MCTS player's pieces are marked as -1, 
        the random actor's pieces as 1, and all other fields as 0.
        """
        single_array_state = np.zeros_like(state[0])
        single_array_state += state[0] * -1
        single_array_state += state[1]
        return single_array_state
