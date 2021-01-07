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
        """
        self.a = kwargs.get("a", 0.5)
        self.b = kwargs.get("b", 0.5)
        self.c = kwargs.get("c", 0.5)
        self.d = kwargs.get("d", 0.5)
        self.r = kwargs.get("r", 1)
        self.s = kwargs.get("s", 1)
        self.x1 = kwargs.get("x1", -1)
        self.x = None
        self.y = None
        self.z = None
        self.current = None
        self.t = None
        self.dt = None
        self.tvec = None

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return (f"HindmarshRose(a={self.a}, b={self.b}, c={self.c}, d={self.d})")

    def _system_equations(self, X, t, current):
        """
        Defines the equations of the dynamical system for integration.
        """
        return [X[1] - self.a * (X[0]**3) + self.b * (X[0]**2) - X[2] + current,
                self.c - self.d * (X[0]**2) - X[1],
                self.r * (self.s * (X[0] - self.x1) - X[2])]

    def run(self, X0=[0, 0, 0], current=1, t=100, dt=0.01):
        """
        Runs the model.

        Args:
            X0 (list, optional): Initial values of x, y and z. Defaults to [0, 0, 0].
            current (int, optional): External current. Defaults to 1.
            t (int, optional): Total time for the simulation. Defaults to 100.
            dt (float, optional): Simulation step. Defaults to 0.01.
        """
        self.current = current
        self.t = t
        self.dt = dt
        self.tvec = np.arange(0, self.t, self.dt)
        X = odeint(self._system_equations, X0, self.tvec, (current,))
        self.x, self.y, self.z = X[:, 0], X[:, 1], X[:, 2]
