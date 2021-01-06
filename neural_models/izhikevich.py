import matplotlib.pyplot as plt
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


def main():
    # Parameters
    t = 200
    current = 10
    Y0 = [-65, 0]

    # Regular Spiking (RS) Neuron
    RS_model = Izhikevich(a=0.02, b=0.2, c=-65, d=8)
    RS_model.run(t, current, Y0)

    # Intrinsically Bursting (RS) Neuron
    IB_model = Izhikevich(a=0.02, b=0.2, c=-55, d=4)
    IB_model.run(t, current, Y0)

    # Chattering (CH) Neuron
    CH_model = Izhikevich(a=0.02, b=0.2, c=-50, d=2)
    CH_model.run(t, current, Y0)

    # Fast Spiking (FS) Neuron
    FS_model = Izhikevich(a=0.1, b=0.2, c=-65, d=2)
    FS_model.run(t, current, Y0)

    # Plots
    fig, ax = plt.subplots(2, 2, figsize=(7, 6))
    ax[0, 0].plot(RS_model.tvec, RS_model.V, 'r')
    ax[0, 0].set_xlabel('time (s)')
    ax[0, 0].set_ylabel('voltage (mv)')
    ax[0, 0].set_title('Regular Spiking (RS)')

    ax[0, 1].plot(IB_model.tvec, IB_model.V, 'g')
    ax[0, 1].set_xlabel('time (s)')
    ax[0, 1].set_ylabel('voltage (mv)')
    ax[0, 1].set_title('Intrinsically Bursting (IB)')

    ax[1, 0].plot(CH_model.tvec, CH_model.V, 'b')
    ax[1, 0].set_xlabel('time (s)')
    ax[1, 0].set_ylabel('voltage (mv)')
    ax[1, 0].set_title('Chattering (CH)')

    ax[1, 1].plot(FS_model.tvec, FS_model.V, 'm')
    ax[1, 1].set_xlabel('time (s)')
    ax[1, 1].set_ylabel('voltage (mv)')
    ax[1, 1].set_title('Fast Spiking (FS)')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()


if __name__ == '__main__':
    main()
