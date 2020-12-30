import numpy as np
import matplotlib.pyplot as plt
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
            I (int, float): External current.
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
        self.I = kwargs.get("I", 1)
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
                "gNa={}, gK={}, gL={})").format(self.C, self.I, self.VNa, self.VK, self.VL,
                                                self.gNa, self.gK, self.gL)

    def system_equations(self, X, t, I):
        """
        Defines the equations of the dynamical system for integration.
        """
        return [(1 / self.C) * (-self.gNa * (X[1] ** 3) * X[3] * (X[0] - self.VNa) - self.gK * (X[2] ** 4) * (X[0] - self.VK) - self.gL * (X[0] - self.VL) + I),
                (0.1 * (X[0] + 40) / (1 - np.exp(-(X[0] + 40) / 10))) * (1 - X[1]) - (4 * np.exp(-(X[0] + 65) / 18)) * X[1],
                (0.01 * (X[0] + 55) / (1 - np.exp(-(X[0] + 55) / 10))) * (1 - X[2]) - (0.125 * np.exp(-(X[0] + 65) / 80)) * X[2],
                (0.07 * np.exp(-(X[0] + 65) / 20)) * (1 - X[3]) - (1 / (1 + np.exp(-(X[0] + 35) / 10))) * X[3]]

    def run(self, X0=[0, 0, 0, 0], I=None):
        """
        Run the model by integrating the system numerically.
        """
        if I is None:
            I = self.I
        else:
            self.I = I
        X = odeint(self.system_equations, X0, self.tvec, (I,))
        self.V, self.m, self.n, self.h = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    def plot(self):
        """
        Plot the membrane potential over time as well as the activation and innactivation of the channels.
        """
        f, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(9, 6))
        f.subplots_adjust(wspace=.3, hspace=.3)
        # Plot V(t)
        ax1.plot(self.tvec, self.V, color='royalblue')
        ax1.set_title("V", fontsize=15)
        ax1.set_ylabel("Voltage [mV]", fontsize=12)
        ax1.grid(alpha=0.3)
        # Plot m(t)
        ax2.plot(self.tvec, self.m, color='lightcoral')
        ax2.set_title("m", fontsize=15)
        ax2.set_ylabel("m(t)", fontsize=12)
        ax2.set_ylim([-0.05, 1.05])
        ax2.grid(alpha=0.3)
        # Plot n(t)
        ax3.plot(self.tvec, self.n, color='goldenrod')
        ax3.set_title("n", fontsize=15)
        ax3.set_xlabel("time [ms]", fontsize=12)
        ax3.set_ylabel("n(t)", fontsize=12)
        ax3.set_ylim([-0.05, 1.05])
        ax3.grid(alpha=0.3)
        # Plot h(t)
        ax4.plot(self.tvec, self.h, color='yellowgreen')
        ax4.set_title("h", fontsize=15)
        ax4.set_xlabel("time [ms]", fontsize=12)
        ax4.set_ylabel("h(t)", fontsize=12)
        ax4.set_ylim([-0.05, 1.05])
        ax4.grid(alpha=0.3)

    def phase_plane(self):
        """
        Plots the phase plane of the system.
        """
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3))
        f.subplots_adjust(wspace=.3)
        # Plot m over Voltage
        ax1.plot(self.V, self.m, color='royalblue')
        ax1.set_xlabel("Voltage [mV]", fontsize=12)
        ax1.set_ylabel("m", fontsize=12)
        ax1.grid(alpha=0.3)
        # Plot n over Voltage
        ax2.plot(self.V, self.n, color='royalblue')
        ax2.set_xlabel("Voltage [mV]", fontsize=12)
        ax2.set_ylabel("n", fontsize=12)
        ax2.grid(alpha=0.3)
        # Plot h over Voltage
        ax3.plot(self.V, self.h, color='royalblue')
        ax3.set_xlabel("Voltage [mV]", fontsize=12)
        ax3.set_ylabel("h", fontsize=12)
        ax3.grid(alpha=0.3)
