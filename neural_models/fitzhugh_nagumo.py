import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class FitzHughNagumo:
    """
    Creates a FitzHugh-Nagumo model.
    """
    def __init__(self, **kwargs):
        """
        Initializes the model.

        Args:
            a (int, float): Variable a.
            b (int, float): Variable b.
            phi (int, float): Variable phi.
            I (int, float): External current.
            dt (int, float): Simulation step.
            t (int): Total time for the simulation.
        """
        self.a = kwargs.get("a", -0.7)
        self.b = kwargs.get("b", 0.8)
        self.I = kwargs.get("I", 0.5)
        self.phi = kwargs.get("phi", 12.5)
        self.dt = kwargs.get("dt", 0.01)
        self.t = kwargs.get("t", 100)

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return ("FitzHughNagumo(a={}, b={}, "
                "I={}, phi={})").format(self.a, self.b, self.I, self.phi)

    def system_equations(self, X, t=0):
        """
        Defines the equations of the dynamical system for integration.
        """
        return [X[0] - (X[0]**3)/3 - X[1] + self.I,
                self.phi*(X[0] + self.a - self.b*X[1])]

    def run(self):
        """
        Run the model by integrating the system numerically.
        """
        X0 = [0, 0]
        self.tvec = np.arange(0, self.t,  self.dt)
        X = odeint(self.system_equations, X0, self.tvec)
        self.V, self.W = X[:, 0], X[:, 1]

    def plot(self):
        """
        Plot the membrane potential over time as well as the recovery variable.
        """
        f, ax1 = plt.subplots(figsize=(8, 5))
        # Plot the potential
        ax1.plot(self.tvec, self.V, color='royalblue')
        ax1.set_xlabel('time [t]', fontsize=12)
        ax1.set_ylabel('Voltage [V]', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='royalblue')
        # Plot the recovery
        ax2 = ax1.twinx()
        ax2.plot(self.tvec, self.W, color='goldenrod')
        ax2.set_ylabel('Recovery [W]', fontsize=12)
        # Within potential limits
        ax2.set_ylim([np.floor(min(self.W)-1), np.ceil(max(self.W))])
        ax2.tick_params(axis='y', labelcolor='goldenrod')
        plt.grid(alpha=0.3)

    def phase_plane(self):
        """
        Plots the phase plane of the system, together with the nullclines.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.V, self.W, color='cornflowerblue')
        plt.plot(self.V, self.V - (self.V**3)/3 + self.I, color="slateblue")
        plt.plot(self.V, (self.V + self.a)/(self.b), color="red")
        plt.xlabel('Voltage [V]', fontsize=12)
        plt.ylabel('Recovery [W]', fontsize=12)
        plt.grid(alpha=0.3)
