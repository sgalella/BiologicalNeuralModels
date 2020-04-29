import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class MorrisLecar(object):
    """
    Creates a MorrisLecar model.
    """
    def __init__(self, **kwargs):
        """
        Initializes the model.

        Args:
            C (int, float): Capacitance of the membrane.
            I (int, float): External current.
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
            dt (int, float): Simulation step.
            time (int): Total time for the simulation.
        """
        self.C = kwargs.get("C", 20)
        self.I = kwargs.get("I", 1)
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
        self.dt = kwargs.get("dt", 0.01)
        self.time = kwargs.get("time", 100)
        self.check_parameters()

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return (f"MorrisLecar(C={self.C}, I={self.I}, VL={self.VL}, VCa={self.VCa}, VK={self.VK}, "
                f"gL={self.gL}), gCa={self.gCa}, gK={self.gK}, V1={self.V1}, V2={self.V2}, "
                f"V3={self.V3}, V4={self.V4}, phi={self.phi}, dt={self.dt}, time={self.time}")

    def check_parameters(self):
        """
        Verify the data types of the inputs
        """
        assert isinstance(self.C, (int, float)), \
            "The capacitance (C) has to be a number."
        assert isinstance(self.I, (int, float)), \
            "The current (I) has to be a number."
        assert isinstance(self.VL, (int, float)), \
            "The potential of L (VL) has to be a number."
        assert isinstance(self.VCa, (int, float)), \
            "The potential of Ca (VCa) a has to be a number."
        assert isinstance(self.VK, (int, float)), \
            "The potential of K (VK) has to be a number."
        assert isinstance(self.gL, (int, float)), \
            "The conductance of L (gL) has to be a number."
        assert isinstance(self.gCa, (int, float)), \
            "The conductance of Ca (gCa) has to be a number."
        assert isinstance(self.gK, (int, float)), \
            "The conductance of K (gK) has to be a number."
        assert isinstance(self.V1, (int, float)), \
            "The potential V1 has to be a number."
        assert isinstance(self.V2, (int, float)), \
            "The potential V2 has to be a number."
        assert isinstance(self.V3, (int, float)), \
            "The potential V3 has to be a number."
        assert isinstance(self.V4, (int, float)), \
            "The potential V4 has to be a number."
        assert isinstance(self.phi, (int, float)), \
            "The phi parameter has to be a number."
        assert isinstance(self.dt, (int, float)), \
            "The step (dt) has to be a number."
        assert isinstance(self.time, (int, float)), \
            "The time has to be a number."

    @property
    def tvec(self):
        """
        Calculates a time vector tvec.
        """
        return np.arange(0, self.time, self.dt)

    def system_equations(self, X, t, I):
        """
        Defines the equations of the dynamical system for integration.
        """
        Mss = (1 + np.tanh((X[0] - self.V1) / self.V2)) / 2
        Nss = (1 + np.tanh((X[0] - self.V3) / self.V4)) / 2
        tau = 1 / self.phi * (np.cosh((X[0] - self.V3) / (2 * self.V4)))

        return [(1 / self.C) * (I - self.gL * (X[0] - self.VL) - self.gCa * Mss * (X[0] - self.VCa) - self.gK * X[1] * (X[0] - self.VK)),
                (Nss - X[1]) / tau]

    def run(self, X0=[0, 0], I=None):
        """
        Run the model by integrating the system numerically.
        """
        if I is None:
            I = self.I
        else:
            self.I = I
        X = odeint(self.system_equations, X0, self.tvec, (I,))
        self.V, self.N = X[:, 0], X[:, 1]

    def plot(self):
        """
        Plot the membrane potential over time as well as the activation and innactivation of the channels.
        """
        f, ([ax1, ax2]) = plt.subplots(1, 2, figsize=(13, 3))
        # Plot V(t)
        ax1.plot(self.tvec, self.V, color='royalblue')
        ax1.set_title("V", fontsize=15)
        ax1.set_ylabel("Voltage [mV]", fontsize=12)
        ax1.grid(alpha=0.3)
        # Plot N(t)
        ax2.plot(self.tvec, self.N, color='lightcoral')
        ax2.set_title("N", fontsize=15)
        ax2.set_ylabel("N(t)", fontsize=12)
        ax2.grid(alpha=0.3)
        plt.show()

    def phase_plane(self):
        """
        Plots the phase plane of the system.
        """
        f = plt.figure(figsize=(7, 4))
        # Plot N over Voltage
        plt.plot(self.V, self.N, color='royalblue')
        plt.xlabel("Voltage [mV]", fontsize=12)
        plt.ylabel("N", fontsize=12)
        plt.grid(alpha=0.3)
        plt.show()
