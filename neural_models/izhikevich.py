from scipy.integrate import solve_ivp


class Izhikevich:
    """ Creates a Izhikevich Model """
    def __init__(self, a, b, c, d):
        """ Initializes the model.

        Args:
            a (float): Time recovery of recovery variable u.
            b (float): Sensitivity of recovery variable u.
            c (float): After-spike reset of membrane potential v.
            d (float): After-spike reset of recovery variable u.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.tvec = None
        self.V = None
        self.U = None

    def __repr__(self):
        """
        Visualize model parameters when printing.
        """
        return f'Izhikevich(a={self.a}, b={self.b}, c={self.c}, d={self.d}'

    def _systems_equations(self, t, X, current):
        """
        Defines the equations of the dynamical system for integration.

        Args:
            t (float): Current time point.
            X (list): Values of membrane potential and recovery variable.
            current (float): External current injected.

        Returns:
            list: Integration of the system of equations.
        """
        if X[0] >= 30:
            X[0] = self.c
            X[1] += self.d
        return [0.04 * X[0] ** 2 + 5 * X[0] + 140 - X[1] + current,
                self.a * (self.b * X[0] - X[1])]

    def run(self, t, current, Y0):
        """
        Run the model by integrating the system numerically.

        Args:
            t (float): End time of integration.
            current (float): External current injected.
            Y0 (list): Initial values of membrane potential and recovery variable.
        """
        tspan = (0, t)
        s = solve_ivp(fun=lambda t, y: self._systems_equations(t, y, current), t_span=tspan, y0=Y0, max_step=0.1)
        self.tvec = s.t
        self.V = s.y[0, :]
        self.U = s.y[1, :]
