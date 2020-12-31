import numpy as np
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
        self.current = kwargs.get("current", 0.5)
        self.phi = kwargs.get("phi", 12.5)
        self.dt = kwargs.get("dt", 0.01)
        self.t = kwargs.get("t", 100)

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return ("FitzHughNagumo(a={}, b={}, "
                "current={}, phi={})").format(self.a, self.b, self.current, self.phi)

    def _system_equations(self, X, t=0):
        """
        Defines the equations of the dynamical system for integration.
        """
        return [X[0] - (X[0]**3) / 3 - X[1] + self.current,
                self.phi * (X[0] + self.a - self.b * X[1])]

    def run(self):
        """
        Run the model by integrating the system numerically.
        """
        X0 = [0, 0]
        self.tvec = np.arange(0, self.t, self.dt)
        X = odeint(self._system_equations, X0, self.tvec)
        self.V, self.W = X[:, 0], X[:, 1]
