import numpy as np


class HopfieldNetwork:
    def __init__(self, n):
        """Initializes the Hopfield network.

        Args:
            n: The number of units
        """
        self.n = n

        self.s = np.random.choice([-1, +1], self.n)

        self.b = np.random.randn(self.n)

        self.w = np.random.randn(self.n, self.n)
        self.w = self.w - np.diag(np.diag(self.w))
        self.w = (self.w + self.w.T) / 2

    def _signals(self, i):
        """Gets the signal input to a specified unit.

        Args:
            i: The index of the unit whose input is calculated

        Returns:
            signals: The signal input to the specified unit
        """
        signals = 0
        for j in range(self.n):
            signals += self.s[j] * self.w[i][j]
        signals += self.b[i]
        return signals

    def update(self, i=None):
        """Updates the state of a specified unit.

        Args:
            i: The index of the unit to update
        """
        if i is None:
            i = np.random.randint(self.n)

        if self._signals(i) < 0:
            self.s[i] = -1
        else:
            self.s[i] = +1

    def is_local_minimum(self):
        """Checks if the Hopfield network is in the local minimum in energy.

        Returns:
            True if the Hopfield network is in the local minimum in energy, otherwise False
        """
        for i in range(self.n):
            if self._signals(i) < 0:
                if self.s[i] != -1:
                    return False
            else:
                if self.s[i] != +1:
                    return False

        return True

    @property
    def energy(self):
        """Gets the energy of the Hopfield network.

        Returns:
            energy: The energy of the Hopfield network
        """
        energy = 0
        for i in range(self.n):
            for j in range(self.n):
                energy -= self.s[i] * self.s[j] * self.w[i][j] / 2
            energy -= self.s[i] * self.b[i]
        return energy
