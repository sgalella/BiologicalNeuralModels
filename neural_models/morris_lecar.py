import numpy as np
from scipy.integrate import odeint


class MorrisLecar:
    """
    Creates a MorrisLecar model.
    """
    def __init__(self, **kwargs):
        """
        Initializes the model.

        Args:
            C (int, float): Capacitance of the membrane.
            VL (int, float): Potential L.
            VCa (int, float): Potential Ca.
            VK (int, float): Potential K.
            gL (int, float): Conductance L.
            gCa (int, float): Conductance Ca.
            gK (int, float): Conductance K.
            V1 (int, float): Potential at which Mss converges.
            V2 (int, float): Reciprocal of slope of Mss.
            V3 (int, float): Potential at which Nss converges.
            V4 (int, float): Reciprocal of slope of Nss.
            phi (int, float): Time scale recovery.
        """
        self.C = kwargs.get("C", 20)
        self.current = kwargs.get("current", 1)
        self.VL = kwargs.get("VL", -60)
        self.VCa = kwargs.get("VCa", 120)
        self.VK = kwargs.get("VK", -84)
        self.gL = kwargs.get("gL", 2)
        self.gCa = kwargs.get("gCa", 4)
        self.gK = kwargs.get("gK", 8)
        self.V1 = kwargs.get("V1", -1.2)
        self.V2 = kwargs.get("V2", 18)
        self.V3 = kwargs.get("V3", 12)
        self.V4 = kwargs.get("V4", 17.4)
        self.phi = kwargs.get("phi", 0.06)
        self.t = None
        self.dt = None
        self.tvec = None
        self.V = None
        self.N = None

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return (f"MorrisLecar(C={self.C}, VL={self.VL}, VCa={self.VCa}, VK={self.VK}, "
                f"gL={self.gL}), gCa={self.gCa}, gK={self.gK}, V1={self.V1}, V2={self.V2}, "
                f"V3={self.V3}, V4={self.V4}, phi={self.phi}")

    def _system_equations(self, X, t, current):
        """
        Defines the equations of the dynamical system for integration.
        """
        Mss = (1 + np.tanh((X[0] - self.V1) / self.V2)) / 2
        Nss = (1 + np.tanh((X[0] - self.V3) / self.V4)) / 2
        tau = 1 / self.phi * (np.cosh((X[0] - self.V3) / (2 * self.V4)))

        return [(1 / self.C) * (current - self.gL * (X[0] - self.VL) - self.gCa * Mss * (X[0] - self.VCa) - self.gK * X[1] * (X[0] - self.VK)),
                (Nss - X[1]) / tau]

    def run(self, X0=[0, 0], current=1, t=100, dt=0.01):
        """
        Runs the model.

        Args:
            X0 (list, optional): Initial values of V and N. Defaults to [0, 0].
            current (int, optional): External current. Defaults to 1.
            t (int, optional): Total time for the simulation. Defaults to 100.
            dt (float, optional): Simulation step. Defaults to 0.01.
        """
        self.current = current
        self.t = t
        self.dt = dt
        self.tvec = np.arange(0, self.t, self.dt)
        X = odeint(self._system_equations, X0, self.tvec, (current,))
        self.V, self.N = X[:, 0], X[:, 1]
