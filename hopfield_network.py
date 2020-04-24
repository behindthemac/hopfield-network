import numpy as np


class HopfieldNetwork:
    def __init__(self, n):
        self.n = n

        self.s = np.random.choice([-1, +1], self.n)

        self.w = np.random.randn(self.n, self.n)
        self.w = self.w - np.diag(np.diag(self.w))
        self.w = (self.w + self.w.T) / 2

        self.b = np.random.randn(self.n)

    @property
    def energy(self):
        energy = 0
        for i in range(self.n):
            for j in range(self.n):
                energy -= self.s[i] * self.s[j] * self.w[i][j] / 2
            energy -= self.s[i] * self.b[i]
        return energy

    def update(self, i=None):
        if i is None:
            i = np.random.randint(self.n)

        if np.dot(self.w[i], self.s) + self.b[i] < 0:
            self.s[i] = -1
        else:
            self.s[i] = +1

    def is_local_minimum(self):
        for i in range(self.n):
            if np.dot(self.w[i], self.s) + self.b[i] < 0:
                if self.s[i] != -1:
                    return False
            else:
                if self.s[i] != +1:
                    return False
        return True
