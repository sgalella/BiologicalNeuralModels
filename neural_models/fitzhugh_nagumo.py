import numpy as np
from scipy.integrate import odeint


class FitzHughNagumo:
    """
    Creates a FitzHugh-Nagumo model.
    """
    def __init__(self, a=-0.7, b=0.8, phi=12.5):
        """
        Initializes the model.

        Args:
            a (int, float): Variable a.
            b (int, float): Variable b.
            phi (int, float): Variable phi.
        """
        self.a = a
        self.b = b
        self.phi = phi
        self.V = None
        self.W = None
        self.current = None
        self.t = None
        self.dt = None
        self.tvec = None

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return f'FitzHughNagumo(a={self.a}, b={self.b}, current={self.current}, phi={self.phi})'

    def _system_equations(self, X, t, current):
        """
        Defines the equations of the dynamical system for integration.
        """
        return [X[0] - (X[0]**3) / 3 - X[1] + current,
                self.phi * (X[0] + self.a - self.b * X[1])]

    def run(self, X0=[0, 0], current=1, t=100, dt=0.01):
        """
        Runs the model.

        Args:
            X0 (list, optional): Initial values of V and W. Defaults to [0, 0, 0].
            current (int, optional): External current. Defaults to 1.
            t (int, optional): Total time for the simulation. Defaults to 100.
            dt (float, optional): Simulation step. Defaults to 0.01.
        """
        self.current = current
        self.t = t
        self.dt = dt
        self.tvec = np.arange(0, self.t, self.dt)
        X = odeint(self._system_equations, X0, self.tvec, (current, ))
        self.V, self.W = X[:, 0], X[:, 1]
