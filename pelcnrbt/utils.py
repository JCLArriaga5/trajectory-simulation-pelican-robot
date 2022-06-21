import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tkinter import PhotoImage
from PIL import Image
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
try:
    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk
except:
    print("Check if you have Gtk installed")

OS = sys.platform

# Pelican Robot Parameters
L1 = L2 = 0.26 # Meters
I1 = 0.1213 # Kg m²
I2 = 0.01616 # Kg m²
LC1 = 0.0983 # meters
LC2 = 0.0229 # meters
M1 = 6.5225 # Kg
M2 = 2.0458 # Kg
GR = 9.80665 # m / s²

def direct_k(q1, q2):
    """
    Direct Kinematic of pelican robot:
        Obtain x-y-cordinates of each link with direct angles

    Parameters
    ----------
    q1 : Direct angle of link_1
    q2 : Direct angle of link_2

    Returns
    -------
    link_1 : x-y-cordinates
    link_2 : x-y-cordinates
    fnl_elmnt : x-y-cordinates of final element of robot
    """

    link_1 = [L1 * np.sin(q1), - L1 * np.cos(q1)]
    link_2 = [L2 * np.sin(q1 + q2), - L2 * np.cos(q1 + q2)]
    fnl_elmnt = [link_1[0] + link_2[0], link_1[1] + link_2[1]]

    return link_1, link_2, fnl_elmnt

def inverse_k(px, py):
    """
    Inverse kinematic of pelican robot:
        Obtain angles for each link to desired position

    Parameters
    ----------
    px : Value of x-position desired
    py : Value of y-position desired

    Returns
    -------
    q1 : Angle in radians of link 1 for the desired position
    q2 : Angle in radians of link 2 for the desired position
    """

    k = (px ** 2 + py ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    q2 = np.arctan2(np.sqrt(1 - (k ** 2)), k)
    q1 = np.arctan2(px, - py) - np.arctan2((L2 * np.sin(q2)), (L1 + L2 * np.cos(q2)))

    return [q1, q2]

class dynamic_model:
    """
    Dynamic model of pelican robot
    ------------------------------
    tau = M(q)\Ddot{q} + C(q, \dot{q})\dot{q} + G(q)
        M(q) := Inertial Matrix
        C(q, \dot{q}) := Centrifugal and Coriolis Forces Matrix
        G(q) := Gravitational Torques Vector
    """

    def M(q2):
        """
        Inertial Matrix

        Parameters
        ----------
        q2 : Angle of link 2
        """

        m_11 = M1 * LC1 ** 2 + M2 * (L1 ** 2 + LC2 ** 2 + 2 * L1 * LC2 * np.cos(q2)) + I1 + I2
        m_12 = M2 * (LC2 ** 2 + L1 * LC2 * np.cos(q2)) + I2
        m_21 = M2 * (LC2 ** 2 + L1 * LC2 * np.cos(q2)) + I2
        m_22 = M2 * LC2 ** 2 + I2

        return [[m_11, m_12], [m_21, m_22]]

    def C(q2, v1, v2):
        """
        Centrifugal and Coriolis Forces Matrix

        Parameters
        ----------
        q2 : Angle of link 2
        v1 : Velocity of link 1
        v2 : Velocity of link 2
        """

        c_11 = - ((M2 * L1 * LC2 * np.sin(q2)) * v2)
        c_12 = - ((M2 * L1 * LC2 * np.sin(q2)) * (v1 + v2))
        c_21 =   ((M2 * L1 * LC2 * np.sin(q2)) * v2)

        return [[c_11, c_12], [c_21, 0.0]]

    def G(q1, q2):
        """
        Gravitational Torques Vector

        Parameters
        ----------
        q1 : Angle of link 1
        q2 : Angle of link 2
        """

        g_11 = (M1 * LC1 + M2 * L1) * GR * np.sin(q1) + M2 * LC2 * GR * np.sin(q1 + q2)
        g_21 = M2 * LC2 * GR * np.sin(q1 + q2)
        return [g_11, g_21]

class binsrch:
    """
    Binary search

    ...

    Attribute
    ---------
    list : Values contained in a list

    Methods
    -------
    def max():
        Maximum list value by binary search

    def min():
        Minimum list value by binary search
    """

    def __init__(self, list):
        """
        Constructor
        """
        self.lst = list
        self.n = len(list) // 2

    def max(self):
        """
        Maximum list value by binary search
        """
        return max(binsrch.__max_value(self.lst[:self.n + 1]),
                   binsrch.__max_value(self.lst[-(self.n + 1):]))

    def min(self):
        """
        Minimum list value by binary search
        """
        return min(binsrch.__min_value(self.lst[:self.n + 1]),
                   binsrch.__min_value(self.lst[-(self.n + 1):]))

    def __max_value(list):
        """
        Recursive maximum value
        """
        if len(list) == 1:
            return list[0]
        else:
            mx = binsrch.__max_value(list[1:])
            return mx if mx > list[0] else list[0]

    def __min_value(list):
        """
        Recursive minimum value
        """
        if len(list) == 1:
            return list[0]
        else:
            mn = binsrch.__min_value(list[1:])
            return mn if mn < list[0] else list[0]

def plot_link(p_i, p_f, *args, **kwargs):
    """
    Function to graph a link in R²

    Parameters
    -----------
    p_i : list [x_i, y_i]
        Initial point of link in R²

    p_f : list [x_f, y_f]
        Final point of link in R²
    """

    plt.plot([p_i[0], p_i[0] + p_f[0]], [p_i[1], p_i[1] + p_f[1]], *args, **kwargs)
    # If the color is not specified, the np.scatter decorator will default to itself.
    if len(args) == 0:
        plt.scatter(p_i[0], p_i[1])
    # Otherwise the color is used regardless of the line style. Since the line
    # style does not attribute to np.sactter.
    else:
        color = ''.join([args[0][i] for i in range(len(args[0]))
                         if args[0][i] not in list(plt.Line2D.lineStyles) + ['.']])
        plt.scatter(p_i[0], p_i[1], facecolor=color)

def set_icon_window(figure_title):
    """
    Change default matplotlib window icon by the pelican robot illustration icon
    and the window title

    Parameters
    ----------
    figure_title : str
        Window tittle
    """
    if OS == 'win32':
        if os.path.exists('../pelcnrbt/images'):
            plc_anim_w = plt.get_current_fig_manager()
            img = PhotoImage(file='images/pelican-robot-icon.png')
            plc_anim_w.window.wm_title(figure_title)
            plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

    elif OS == 'linux' or 'darwin':
        if os.path.exists('../pelcnrbt/images'):
            plc_anim_w = plt.get_current_fig_manager()
            plc_anim_w.window.set_title(figure_title)
            plc_anim_w.window.set_icon_from_file(filename='images/pelican-robot-icon.png')

def message_dialog(title, message):
    if OS == 'linux':
        dialog = Gtk.MessageDialog(
           transient_for=None,
           flags=0,
           message_type=Gtk.MessageType.INFO,
           buttons=Gtk.ButtonsType.OK,
           text=title,
        )
        dialog.format_secondary_text(message)
        dialog.run()

        dialog.destroy()

class realtime:
    """
    Generate animation of the trajectory and how it was reducing the qt error
    until reaching the desired position.

    ...

    Methods
    -------
    def show(ts, qs, qt, dp):
        Once each of the necessary values for the animation are obtained, it is
        shown in the established time interval as it was changing.
    """

    def __init__(self):
        """
        Constructor

        In the constructor the shape of the figure is initialized and designed
        for animation.
        """

        # Number of subplots to animate
        self.fig, (self.robot, self.qt) = plt.subplots(2, 1)
        # Change icon window & figire tittle
        set_icon_window('Pelican Robot: Trajectory Simulation')
        # Limits and appearance of the pelican robot animation
        self.robot.set_xlim((-0.6, 0.6))
        self.robot.set_ylim((-0.6, 0.6))
        self.robot.set_xticks(np.arange(-0.6, 0.6, 0.1))
        self.robot.set_yticks(np.arange(-0.6, 0.6, 0.1))
        self.robot.grid(which='both')
        self.robot.set_xticklabels([])
        self.robot.set_yticklabels([])
        self.robot.set_aspect('equal', adjustable='box')

        # Some elements for the animation of the error qt
        self.qt.set_title("Graph of how the error ($ \\tilde{q} $) was reduced.")
        self.qt.set_xlabel("$ t(s) $", fontsize='large')
        self.qt.set_ylabel("$ \\tilde{q} $", rotation='horizontal', fontsize='large')
        self.qt.grid('on', linestyle='--')

    def show(self, ts, qs, qt, dp, ctrl_type):
        """
        Function to generate the animation using the parameters necessary for
        the animation.

        Parameters
        ----------
        ts : List with the integration step intervals for the set time range.
        qs : List with the values of the angles of each link of each iteration
             until reaching the desired position.
        qt : List with the values of how the error qt was changing in each
             iteration until reaching the desired position.
        dp : Desired position in the form [Px, Py] to draw the point the robot
             should reach in the animation.
        ctrl_type : str
            PD := Proportional control plus velocity feedback and Proportional
                  Derivative(PD) control
            PD-GC := PD control with gravity compensation
            PD-dGC := PD control with desired gravity compensation
        """

        print('Realtime animation for desired position : [{}, {}], Start!'.format(dp[0], dp[1]))

        # Get error for each link
        q1e = [qt[n][0] for n in range(len(qt))]
        q2e = [qt[n][1] for n in range(len(qt))]
        # Get limits of qt plot
        self.qt.set_xlim((binsrch(ts).min(), binsrch(ts).max()))
        qtmin = min(binsrch(q1e).min(), binsrch(q2e).min())
        qtmax = max(binsrch(q1e).max(), binsrch(q2e).max())
        self.qt.set_ylim((qtmin, qtmax))
        # Text for formulas in qt plot
        if ctrl_type == 'PD':
            str_controller = r'$ \tau = K_{p} \tilde{q} - K_{v} \dot{q} $'
        elif ctrl_type == 'PD-GC':
            str_controller = r'$ \tau = K_{p} \tilde{q} - K_{v} \dot{q} + g(q) $'
        elif ctrl_type == 'PD-dGC':
            str_controller = r'$ \tau = K_{p} \tilde{q} - K_{v} \dot{q} + g(q_d) $'

        str_qt = r'$ \tilde{q} := q_d - q(t) $'
        self.qt.text((max(ts) / 2), (qtmax), 'Controller equation',
                     horizontalalignment='center', verticalalignment='top')
        self.qt.text((max(ts) / 2), (qtmax - 200 * 0.001), str_controller,
                     horizontalalignment='center', verticalalignment='top')
        self.qt.text((max(ts) / 2), (qtmax - 500 * 0.001), str_qt, horizontalalignment='center',
                     verticalalignment='top')
        # Some elements for the animation of the pelican robot
        self.robot.set_title(r"Pelican Robot: Trajectory to [{}, {}]".format(dp[0], dp[1]))
        self.robot.scatter(dp[0], dp[1], facecolor='k')
        # For pelican robot animation
        link_1, = self.robot.plot([], [], 'r', lw=2)
        link_2, = self.robot.plot([], [], 'b', lw=2)
        # For qt animation
        q1_error, = self.qt.plot([], [], 'r', lw=2, label="$ \\tilde{q_{1}} $")
        q2_error, = self.qt.plot([], [], 'b', lw=2, label="$ \\tilde{q_{2}} $")
        self.qt.legend(loc='upper right')

        def __animate(i):
            """
            Generate animation
            """

            if i == len(qs) - 1:
                print('Realtime animation done!, Please close window.')
                # plt.close(self.fig)

                return link_1, link_2, q1_error, q2_error,

            else:
                # print(f'frame: {i}') # Debug: May be useful to stop
                link_1.set_data([0.0, direct_k(qs[i][0], qs[i][1])[0][0]],
                                [0.0, direct_k(qs[i][0], qs[i][1])[0][1]])

                link_2.set_data([direct_k(qs[i][0], qs[i][1])[0][0],
                                 direct_k(qs[i][0], qs[i][1])[2][0]],
                                [direct_k(qs[i][0], qs[i][1])[0][1],
                                 direct_k(qs[i][0], qs[i][1])[2][1]])

                q1_error.set_data(ts[:i - len(ts)], q1e[:i - len(q1e)])
                q2_error.set_data(ts[:i - len(ts)], q2e[:i - len(q2e)])

                return link_1, link_2, q1_error, q2_error,

        ani = FuncAnimation(self.fig, __animate, interval=1, blit=True, frames=len(qs),
                            repeat=False)

        self.fig.tight_layout()
        plt.show()
