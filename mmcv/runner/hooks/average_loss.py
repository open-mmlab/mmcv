class AverageLoss:
    def __init__(self, name=None, n=500):
        assert n > 0
        self.name = name
        self.window_len = n
        self.reset()

    def reset(self):
        self.loss_history = list()
        self.count = 0

    def update(self, val):
        assert isinstance(val, (float, int))
        self.loss_history.append(val)
        self.count += 1

    def average(self):
        history = self.loss_history[-self.window_len:]
        return sum(history) / len(history)