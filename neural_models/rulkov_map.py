import numpy as np
import matplotlib.pyplot as plt


class RulkovMap(object):
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
        self.check_parameters()  # Check if the inputs have the correct format

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

    def plot(self):
        """
        Plot the membrane potential over time as well as the slow dynamics variable.
        """
        f, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(self.tvec, self.x, color="royalblue")
        ax1.set_xlabel('time [t]', fontsize=12)
        ax1.set_ylabel(r"$x_n$", fontsize=15)
        ax1.tick_params(axis='y', labelcolor='royalblue')
        plt.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(self.tvec, self.y, color="goldenrod")
        ax2.set_ylabel(r"$y_n$", fontsize=15)
        ax2.tick_params(axis='y', labelcolor='goldenrod')
