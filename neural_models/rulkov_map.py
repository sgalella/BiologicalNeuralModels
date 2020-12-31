import numpy as np


class RulkovMap:
    """
    Creates a Rulkov Map model.
    """
    def __init__(self, **kwargs):
        """
        Initializes the model.

        Args:
            alpha (int, float): Variable alpha.
            mu (int, float): Variable mu.
            sigma (int, float): Variable sigma.
            time (int, float): Total time for the simulation.
            x0 (int, float): Intial value for the membrane potential.
            y0 (int, float): Initial value for the slow dynamics.
        """
        self.alpha = kwargs.get("alpha", 6)
        self.mu = kwargs.get("mu", 0.1)
        self.sigma = kwargs.get("sigma", 0.3)
        self.time = kwargs.get("time", 100)
        self.x0 = kwargs.get("x0", 0)
        self.y0 = kwargs.get("x0", 0)

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return (f"RulkovModel(alpha={self.alpha}, mu={self.mu}, sigma={self.sigma}, t={self.time})")

    @property
    def tvec(self):
        """
        Calculates a time vector tvec.
        """
        return np.arange(0, self.time, 1)

    def run(self):
        """
        Runs the model.
        """
        self.x = np.zeros(self.tvec.shape)
        self.y = np.zeros(self.tvec.shape)
        self.x[0] = self.x0
        self.y[0] = self.y0
        for t in range(1, self.time - 1):
            if self.x[t - 1] <= 0:
                self.x[t] = (self.alpha / (1 - self.x[t - 1])) + self.y[t - 1]
            elif 0 < self.x[t - 1] < self.alpha + self.y[t - 1]:
                self.x[t] = self.alpha + self.y[t - 1]
            elif self.x[t - 1] >= self.alpha + self.y[t - 1]:
                self.x[t] = -1
            self.y[t] = self.y[t - 1] - self.mu * (self.x[t - 1] + 1) + self.mu * self.sigma
