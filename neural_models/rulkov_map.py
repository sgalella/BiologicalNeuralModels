import numpy as np


class RulkovMap:
    """
    Creates a Rulkov Map model.
    """
    def __init__(self, alpha=6, mu=0.1, sigma=0.3, x0=0, y0=0):
        """
        Initializes the model.

        Args:
            alpha (int, float): Variable alpha.
            mu (int, float): Variable mu.
            sigma (int, float): Variable sigma.
            x0 (int, float): Intial value for the membrane potential.
            y0 (int, float): Initial value for the slow dynamics.
        """
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
        self.y0 = y0
        self.x = None
        self.y = None
        self.t = None
        self.tvec = None

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return (f"RulkovModel(alpha={self.alpha}, mu={self.mu}, sigma={self.sigma})")

    def run(self, t=100):
        """
        Run the model by integrating the system numerically.

        Args:
            t (int, optional): Total time for the simulation. Defaults to 100.
        """
        self.t = t
        self.tvec = np.arange(0, self.t)
        self.x = np.zeros(self.tvec.shape)
        self.y = np.zeros(self.tvec.shape)
        self.x[0] = self.x0
        self.y[0] = self.y0
        for t in self.tvec:
            if self.x[t - 1] <= 0:
                self.x[t] = (self.alpha / (1 - self.x[t - 1])) + self.y[t - 1]
            elif 0 < self.x[t - 1] < self.alpha + self.y[t - 1]:
                self.x[t] = self.alpha + self.y[t - 1]
            elif self.x[t - 1] >= self.alpha + self.y[t - 1]:
                self.x[t] = -1
            self.y[t] = self.y[t - 1] - self.mu * (self.x[t - 1] + 1) + self.mu * self.sigma
