import numpy as np
from scipy.integrate import odeint


class HodgkinHuxley:
    """
    Creates a HodgkinHuxley model.
    """
    def __init__(self, C=1, VNa=50, VK=-77, VL=-54.4, gNa=120, gK=36, gL=0.3):
        """
        Initializes the model.

        Args:
            C (int, float): Capacitance of the membrane.
            VNa (int, float): Potential Na.
            VK (int, float): Potential K.
            VL (int, float): Potential L.
            gNa (int, float): Conductance Na.
            gK (int, float): Conductance K.
            gL (int, float): Conductance L.
        """
        self.C = C
        self.VNa = VNa
        self.VK = VK
        self.VL = VL
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.V = None
        self.m = None
        self.n = None
        self.h = None

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return ("HodgkinHuxley(C={}, "
                "VNa={}, VK={}, VL={} "
                "gNa={}, gK={}, gL={})").format(self.C, self.VNa, self.VK, self.VL,
                                                self.gNa, self.gK, self.gL)

    def _system_equations(self, X, t, current):
        """
        Defines the equations of the dynamical system for integration.
        """
        return [(1 / self.C) * (-self.gNa * (X[1] ** 3) * X[3] * (X[0] - self.VNa) - self.gK * (X[2] ** 4) * (X[0] - self.VK) - self.gL * (X[0] - self.VL) + current),
                (0.1 * (X[0] + 40) / (1 - np.exp(-(X[0] + 40) / 10))) * (1 - X[1]) - (4 * np.exp(-(X[0] + 65) / 18)) * X[1],
                (0.01 * (X[0] + 55) / (1 - np.exp(-(X[0] + 55) / 10))) * (1 - X[2]) - (0.125 * np.exp(-(X[0] + 65) / 80)) * X[2],
                (0.07 * np.exp(-(X[0] + 65) / 20)) * (1 - X[3]) - (1 / (1 + np.exp(-(X[0] + 35) / 10))) * X[3]]

    def run(self, X0=[0, 0, 0, 0], current=1, t=100, dt=0.01):
        """
        Runs the model.

        Args:
            X0 (list, optional): Initial values of V, m, n, and h. Defaults to [0, 0, 0, 0].
            current (int, optional): External current. Defaults to 1.
            t (int, optional): Total time for the simulation. Defaults to 100.
            dt (float, optional): Simulation step. Defaults to 0.01.
        """
        self.current = current
        self.t = t
        self.dt = dt
        self.tvec = np.arange(0, self.t, self.dt)
        X = odeint(self._system_equations, X0, self.tvec, (current,))
        self.V, self.m, self.n, self.h = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
