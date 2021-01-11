import numpy as np


class LeakyIntegrateAndFire:
    """
    Creates an integrate-and-fire model.
    """
    def __init__(self, VR=-70, R=100, C=0.3, theta=-55):
        """
        Initializes the model.
        Args:
            VR (int, float): Resting state potential.
            R (int, float): Resistance of the cell membrane.
            C (int, float): Capacitance of the cell membrane.
        """
        self.VR = VR
        self.R = R
        self.C = C
        self.theta = theta
        self.t = None
        self.dt = None
        self.tvec = None
        self.V = None

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return ("LeakyIntegrateAndFire(VR={}, R={}, C={}, "
                "theta={}").format(self.VR, self.R, self.C, self.theta)

    def run(self, current=1, t=100, dt=0.01):
        """
        Runs the model.

        Args:
            current (int, optional): External current. Defaults to 1.
            t (int, optional): Total time for the simulation. Defaults to 100.
            dt (float, optional): Simulation step. Defaults to 0.01.
        """
        self.current = current
        self.t = t
        self.dt = dt
        self.tvec = np.arange(0, self.t, self.dt)
        self.tau = self.R * self.C
        self.V = np.zeros(self.tvec.shape)
        step = 0
        for idx in range(len(self.tvec)):
            self.V[idx] = self.VR + self.R * self.current * (1 - np.exp(-step / (self.tau)))
            step += self.dt
            if self.V[idx] > self.theta:
                step = 0
