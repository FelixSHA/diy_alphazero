class DeepQ:
      def __init__(self, model, optimizer, game, args):
            self.model = model
            self.optimizer = optimizer
            self.game = game
            self.args = args