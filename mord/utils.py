import numpy as np

class OverfittingCV:
    def __init__(self, n, n_iter):
        self.n = n
        self.n_iter = n_iter
    
    def __iter__(self):
        for i in range(self.n_iter):
            train = np.arange(self.n)
            test = train
            yield train, test

