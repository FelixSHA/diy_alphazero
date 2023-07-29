import numpy as np

# Define the Node class which will be used in MCTS
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.rewards = 0
        self.unvisited_actions = list(range(self.state.shape[1]))  # initially all actions are unvisited
        self.action = action  # action that led to this state

# Define the MCTS class
class MCTS:
    def __init__(self, game, exploration_weight=1):
        self.game = game
        self.exploration_weight = exploration_weight
        self.root = Node(game.get_initial_state())
        self.node_lookup = {}

    def uct(self, node):
        """
        Calculate UCT value for a node.
        """
        if node.visits == 0:
            return np.inf  # encourage exploration of unvisited nodes
        else:
            return node.rewards / node.visits + self.exploration_weight * np.sqrt(
                2 * np.log(node.parent.visits) / node.visits)

    def select(self, node):
        """
        Selection phase of MCTS.
        Traverse the tree until we reach a leaf node.
        """
        while len(node.unvisited_actions) == 0 and len(node.children) > 0:
            node = max(node.children.values(), key=self.uct)
        return node

    def expand(self, node):
        """
        Expansion phase of MCTS.
        Add a new child for the current node.
        """
        valid_moves = self.game.get_valid_moves(node.state)
        node.unvisited_actions = [action for action in node.unvisited_actions if valid_moves[action] == 1]
        if len(node.unvisited_actions) == 0:
            return node  # No valid moves left, return the current node
        action = node.unvisited_actions.pop()
        next_state = self.game.get_next_state(node.state.copy(), action, 1)
        child_node = Node(next_state, parent=node, action=action)
        node.children[action] = child_node
        return child_node


    def simulate(self, node):
        """
        Simulation phase of MCTS.
        Play out a random game from the current node.
        """
        player = 1
        state = node.state.copy()
        while True:
            valid_moves = self.game.get_valid_moves(state)
            if valid_moves.sum() == 0:
                return 0  # draw
            action = np.random.choice(valid_moves.nonzero()[0])
            state = self.game.get_next_state(state, action, player)
            if self.game.check_win(state, action):
                return player
            player = self.game.get_opponent(player)

    def backpropagate(self, node, reward):
        """
        Backpropagation phase of MCTS.
        Propagate the results of the simulation back up the tree.
        """
        while node is not None:
            node.visits += 1
            node.rewards += reward
            node = node.parent

    def run(self, num_iterations):
        """
        Run the MCTS algorithm for a certain number of iterations.
        """
        for _ in range(num_iterations):
            leaf_node = self.select(self.root)
            if leaf_node.visits == 0 or len(leaf_node.unvisited_actions) > 0:
                new_node = self.expand(leaf_node)
                reward = self.simulate(new_node)
                self.backpropagate(new_node, reward)
            else:
                reward = 0 if leaf_node.rewards == 0 else 1 if leaf_node.rewards > 0 else -1
                self.backpropagate(leaf_node, reward)

    def get_best_move(self):
        """
        Return the best move according to the MCTS.
        """
        return max(self.root.children.values(), key=lambda node: node.visits).action
    
    def run(self, state, num_iterations):
        """
        Run the MCTS algorithm for a certain number of iterations.
        """
        self.root = Node(state)
        for _ in range(num_iterations):
            leaf_node = self.select(self.root)
            if leaf_node.visits == 0 or len(leaf_node.unvisited_actions) > 0:
                new_node = self.expand(leaf_node)
                reward = self.simulate(new_node)
                self.backpropagate(new_node, reward)
            else:
                reward = 0 if leaf_node.rewards == 0 else 1 if leaf_node.rewards > 0 else -1
                self.backpropagate(leaf_node, reward)
