import numpy as np
from scipy.integrate import odeint


class HodgkinHuxley:
    """
    Creates a HodgkinHuxley model.
    """
    def __init__(self, **kwargs):
        """
        Initializes the model.

        Args:
            C (int, float): Capacitance of the membrane.
            current (int, float): External current.
            VNa (int, float): Potential Na.
            VK (int, float): Potential K.
            VL (int, float): Potential L.
            gNa (int, float): Conductance Na.
            gK (int, float): Conductance K.
            gL (int, float): Conductance L.
            dt (int, float): Simulation step.
            t (int): Total time for the simulation.
        """
        self.C = kwargs.get("C", 1)
        self.current = kwargs.get("current", 1)
        self.VNa = kwargs.get("VNa", 50)
        self.VK = kwargs.get("VK", -77)
        self.VL = kwargs.get("VL", -54.4)
        self.gNa = kwargs.get("gNa", 120)
        self.gK = kwargs.get("gK", 36)
        self.gL = kwargs.get("gL", 0.3)
        self.dt = kwargs.get("dt", 0.01)
        self.t = kwargs.get("t", 100)
        self.tvec = np.arange(0, self.t, self.dt)

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return ("HodgkinHuxley(C={}, I={}, "
                "VNa={}, VK={}, VL={} "
                "gNa={}, gK={}, gL={})").format(self.C, self.current, self.VNa, self.VK, self.VL,
                                                self.gNa, self.gK, self.gL)

    def _system_equations(self, X, t, current):
        """
        Defines the equations of the dynamical system for integration.
        """
        return [(1 / self.C) * (-self.gNa * (X[1] ** 3) * X[3] * (X[0] - self.VNa) - self.gK * (X[2] ** 4) * (X[0] - self.VK) - self.gL * (X[0] - self.VL) + current),
                (0.1 * (X[0] + 40) / (1 - np.exp(-(X[0] + 40) / 10))) * (1 - X[1]) - (4 * np.exp(-(X[0] + 65) / 18)) * X[1],
                (0.01 * (X[0] + 55) / (1 - np.exp(-(X[0] + 55) / 10))) * (1 - X[2]) - (0.125 * np.exp(-(X[0] + 65) / 80)) * X[2],
                (0.07 * np.exp(-(X[0] + 65) / 20)) * (1 - X[3]) - (1 / (1 + np.exp(-(X[0] + 35) / 10))) * X[3]]

    def run(self, X0=[0, 0, 0, 0], current=None):
        """
        Run the model by integrating the system numerically.
        """
        if current is None:
            current = self.current
        else:
            self.current = current
        X = odeint(self._system_equations, X0, self.tvec, (current,))
        self.V, self.m, self.n, self.h = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
