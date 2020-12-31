import numpy as np
from scipy.integrate import odeint


class HindmarshRose:
    """
    Creates a HindmarshRose model.
    """
    def __init__(self, **kwargs):
        """
        Initializes the model.

        Args:
            current (int, float): External current.
            a (int, float): Positive parameter.
            b (int, float): Positive parameter.
            c (int, float): Positive parameter.
            d (int, float): Positive parameter.
            r (int, float): Positive parameter.
            s (int, float): Positive parameter.
            x1 (int, float): Resting potential.
            dt (int, float): Simulation step.
            time (int): Total time for the simulation.
        """
        self.current = kwargs.get("current", 1)
        self.a = kwargs.get("a", 0.5)
        self.b = kwargs.get("b", 0.5)
        self.c = kwargs.get("c", 0.5)
        self.d = kwargs.get("d", 0.5)
        self.r = kwargs.get("r", 1)
        self.s = kwargs.get("s", 1)
        self.x1 = kwargs.get("x1", -1)
        self.dt = kwargs.get("dt", 0.01)
        self.time = kwargs.get("time", 100)

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return (f"HindmarshRose(a={self.a}, b={self.b}, c={self.c}, d={self.d}, dt={self.dt}, time={self.time})")

    @property
    def tvec(self):
        """
        Calculates a time vector tvec.
        """
        return np.arange(0, self.time, self.dt)

    def _system_equations(self, X, t, current):
        """
        Defines the equations of the dynamical system for integration.
        """
        return [X[1] - self.a * (X[0]**3) + self.b * (X[0]**2) - X[2] + current,
                self.c - self.d * (X[0]**2) - X[1],
                self.r * (self.s * (X[0] - self.x1) - X[2])]

    def run(self, X0=[0, 0, 0], current=None):
        """
        Run the model by integrating the system numerically.
        """
        if current is None:
            current = self.current
        else:
            self.current = current
        X = odeint(self._system_equations, X0, self.tvec, (current,))
        self.x, self.y, self.z = X[:, 0], X[:, 1], X[:, 2]
