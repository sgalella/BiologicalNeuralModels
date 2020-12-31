import numpy as np


class LeakyIntegrateAndFire:
    """
    Creates an integrate-and-fire model.
    """
    def __init__(self, **kwargs):
        """
        Initializes the model.
        Args:
            VR (int, float): Resting state potential.
            R (int, float): Resistance of the cell membrane.
            C (int, float): Capacitance of the cell membrane.
            current (int, float): External current.
            dt (int, float): Simulation step.
            t (int): Total time for the simulation.
            threshold
        """
        self.VR = kwargs.get("VR", -70)
        self.R = kwargs.get("R", 100)
        self.C = kwargs.get("C", 0.3)
        self.current = kwargs.get("current", 0.3)
        self.theta = kwargs.get("theta", -55)
        self.dt = kwargs.get("dt", 0.01)
        self.t = kwargs.get("t", 100)

    @property
    def tau(self):
        """
        Calculates tau given R and C.
        """
        return self.R * self.C

    @property
    def tvec(self):
        """
        Calculates a time vector tvec.
        """
        return np.arange(0, self.t, self.dt)

    @property
    def period(self):
        """
        Calculates the period given the model parameters.
        """
        return - self.tau * np.log(1 - (self.theta - self.VR) / (self.R * self.current))

    @property
    def frequency(self):
        """
        Calculates the frequency given the period.
        """
        return 1 / self.period

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return ("LeakyIntegrateAndFire(VR={}, R={}, C={}, tau={}, "
                "I={}, theta={}, dt={}, t={}, "
                "T={}, f={})").format(self.VR, self.R, self.C, self.tau,
                                      self.current, self.theta, self.dt, self.t,
                                      round(self.period, 3),
                                      round(self.frequency, 3))

    def run(self):
        """
        Run the model.
        """
        self.V = np.zeros(self.tvec.shape)
        step = 0
        for idx in range(len(self.tvec)):
            self.V[idx] = self.VR + self.R * self.current * (1 - np.exp(-step / (self.tau)))
            step += self.dt
            if self.V[idx] > self.theta:
                step = 0

