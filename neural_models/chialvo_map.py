import numpy as np


class ChialvoMap:
    """
    Creates a Chialvo Map model.
    """
    def __init__(self, a, b, c, k, x0=0, y0=0):
        """
        Initializes the model.

        Args:
            k (int, float): Variable alpha.
            a (int, float): Variable mu.
            b (int, float): Variable sigma.
            c (int, float): Intial value for the membrane potential.
        """
        self.a = a
        self.b = b
        self.c = c
        self.k = k
        self.x0 = x0
        self.y0 = y0
        self.t = None
        self.tvec = None

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return f'ChialvoMap(a={self.a}, b={self.b}, c={self.c}, k={self.k})'

    def run(self, X0=[0, 0], t=100):
        """
        Run the model by integrating the system numerically.

        Args:
            t (int, optional): Total time for the simulation. Defaults to 100.
        """
        self.t = t
        self.tvec = np.arange(0, self.t + 1)
        self.x = np.zeros(self.tvec.shape)
        self.y = np.zeros(self.tvec.shape)
        self.x[0] = X0[0]
        self.y[0] = X0[1]
        for t in range(0, self.t - 1):
            self.x[t + 1] = (self.x[t] ** 2) * np.exp(self.y[t] - self.x[t]) + self.k
            self.y[t + 1] = self.a * self.y[t] - self.b * self.x[t] + self.c
