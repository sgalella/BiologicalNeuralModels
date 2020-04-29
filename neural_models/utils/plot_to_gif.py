import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
# plt.style.use('seaborn-pastel')


def initialize_animation():
    line.set_data([], [])
    return line,

def create_animation(i):
    I = i - 5   # i is the frame! Here going from -5 to 20
    model.run([-65, 0, 0, 0], I)
    x = model.tvec
    y = model.V
    line.set_data(x, y)
    ax.set_title("Current = {} [nA]".format(I))
    return line,

# Figure
fig = plt.figure()
ax = plt.axes(xlim=(0,100), ylim=(-80, 40))
line, = ax.plot([], [], color="royalblue")
ax.set_xlabel("time [ms]", fontsize=12)
ax.set_ylabel("Voltage [mV]", fontsize=12)
animation = FuncAnimation(fig, create_animation, init_func=initialize_animation,
                          frames=16, interval=170, blit=True)
animation.save('../../images/hodgkin-huxley.gif', writer='imagemagick')