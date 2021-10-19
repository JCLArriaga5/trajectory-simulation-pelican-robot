#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Trajectory simulation of the Pelican Prototype Robot of the CICESE Research
# Center, Mexico. The position control used was PD control with gravity
# compensation and the 4th order Runge-Kutta method.
#
# References
#   Kelly, R., Davila, V. S., & Perez, J. A. L. (2006). Control of robot
#   manipulators in joint space. Springer Science & Business Media.

import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import PhotoImage
from PIL import Image
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Pelican Robot Parameters
L1 = L2 = 0.26 # Meters
I1 = 0.1213 # Kg m²
I2 = 0.01616 # Kg m²
LC1 = 0.0983 # meters
LC2 = 0.0229 # meters
M1 = 6.5225 # Kg
M2 = 2.0458 # Kg
GR = 9.80665 # m / s²

class binsrch:
    """
    Binary search

    ...

    Attribute
    ---------
    list : Values contained in a list

    Methods
    -------
    def __max__():
        Maximum list value by binary search

    def __min__():
        Minimum list value by binary search
    """

    def __init__(self, list):
        """
        Constructor
        """
        self.lst = list
        self.n = len(list) // 2

    def __max__(self):
        """
        Maximum list value by binary search
        """
        return max(binsrch.max_value(self.lst[:self.n + 1]),
                   binsrch.max_value(self.lst[-(self.n + 1):]))

    def __min__(self):
        """
        Minimum list value by binary search
        """
        return min(binsrch.min_value(self.lst[:self.n + 1]),
                   binsrch.min_value(self.lst[-(self.n + 1):]))

    @staticmethod
    def max_value(list):
        """
        Recursive maximum value
        """
        if len(list) == 1:
            return list[0]
        else:
            mx = binsrch.max_value(list[1:])
            return mx if mx > list[0] else list[0]

    @staticmethod
    def min_value(list):
        """
        Recursive minimum value
        """
        if len(list) == 1:
            return list[0]
        else:
            mn = binsrch.min_value(list[1:])
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

    K = (px ** 2 + py ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    q2 = np.arctan2(np.sqrt(1 - (K ** 2)), K)
    q1 = np.arctan2(px, - py) - np.arctan2((L2 * np.sin(q2)), (L1 + L2 * np.cos(q2)))

    return [q1, q2]

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
        # Change window title
        # self.fig.canvas.set_window_title('Pelican Robot: Simulation')
        # Change icon window
        if os.path.exists('../pelcnrbt/images'):
            plc_anim_w = plt.get_current_fig_manager()
            img = PhotoImage(file='images/pelican-robot-icon.png')
            plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

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
        """

        print('Realtime animation for desired position : [{}, {}], Start!'.format(dp[0], dp[1]))

        # Get error for each link
        q1e = [qt[n][0] for n in range(len(qt))]
        q2e = [qt[n][1] for n in range(len(qt))]
        # Get limits of qt plot
        self.qt.set_xlim((binsrch(ts).__min__(), binsrch(ts).__max__()))
        qtmin = min(binsrch(q1e).__min__(), binsrch(q2e).__min__())
        qtmax = max(binsrch(q1e).__max__(), binsrch(q2e).__max__())
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

        def animate(i):
            """
            Generate animation
            """

            if i == len(qs) - 1:
                print('Realtime animation donde!, Closed window.')
                plt.close('all')

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

        ani = FuncAnimation(self.fig, animate, interval=5, blit=True, frames=len(qs),
                            repeat=False)

        self.fig.tight_layout()
        plt.show()

class pelican_robot:
    """
    Simulate a desired potion for the pelican robot with position control

    ...

    Attributes
    ----------
    dp : list
        Desired position [Px, Py]
    kp : list array
        Position gain in matrix form, e.g. [[30.0, 0.0],[0.0, 30.0]].
    kv : list array
        Velocity gain in matrix form, e.g. [[7.0, 0.0],[0.0, 2.0]].
    **kwarg : Controller type, default is PD
        control_law = {'PD', 'PD-GC', 'PD-dGC'}
        PD := Proportional control plus velocity feedback and Proportional
              Derivative(PD) control
        PD-GC := PD control with gravity compensation
        PD-dGC := PD control with desired gravity compensation

    Methods
    -------
    RK4(ti, ui, vi, tf, h, display):
        Runge-Kutta 4th order to solve system ODE's to obtain angles and velocities.

    controller(qs, qps):
        - Controller that returns the error of each angle of the desired position.
        - Generate the controller tau with the set gains to use within the class.

    BCG(q, v):
        Canonical form of the robot's Lagrange equation of motion, the integrated
        controller with gravitational compensation. Return the accelerations of
        each link.

    plot_velocity_bhvr():
        Function to graph the behavior of the speed of each iteration after RK4
        is calculated first.

    plot_q_error():
        Function to graph the behavior of how the error of each link was reduced
        until reaching the desired position, after RK4 is calculated first.

    plot_trajectory(stp):
        Function to graph trajectory with each angle iteration to desired position.

    get_traj_gif(fname, duration):
        Function to save GIF of trajectory to desired point

    values():
        Function to return values of each iteration in RK4.
    """

    def __init__(self, dp, kp, kv, **kwarg):
        """
        Constructor

        Parameters
        ----------
        dp : Desired position ([Px, Py] in R²)
        kp : Position gain. e.g. [[30.0, 0.0],[0.0, 30.0]]
        kv : Velocity gain. e.g. [[7.0, 0.0],[0.0, 2.0]]
        **kwarg : Controller type, default is PD
            control_law = {'PD', 'PD-GC', 'PD-dGC'}
            PD := Proportional control plus velocity feedback and Proportional
                  Derivative(PD) control
            PD-GC := PD control with gravity compensation
            PD-dGC := PD control with desired gravity compensation
        """

        if pelican_robot.pts_in_range(dp):
            self.dp = dp
        else:
            raise ValueError('{} not in range of workspace robot'.format(str(dp)))
        if len(kp) != 2:
            raise ValueError('{} Cannot have more than two lists within itself'.format('kp = ' + str(kp)))
        if all([True if len(kp[i]) == 2 else False for i in range(len(kp))]) is not False:
            self.kp = kp
        else:
            raise ValueError(""" Inner lists should only contain 2 values like:
                kp = [[gain_value, 0.0], [0.0, gain_value]] """)
        if len(kv) != 2:
            raise ValueError('{} Cannot have more than two lists within itself'.format('kv = ' + str(kv)))
        if all([True if len(kv[i]) == 2 else False for i in range(len(kv))]) is not False:
            self.kv = kv
        else:
            raise ValueError(""" Inner lists should only contain 2 values like:
                kv = [[gain_value, 0.0], [0.0, gain_value]] """)
        self.ctrl_type = pelican_robot.chk_kwarg(kwarg)
        self.ts = []
        self.qs = []
        self.vs = []
        self.qerr = []
        self.tau = [0.0, 0.0]

    @staticmethod
    def pts_in_range(point):
        """
        Returns True if the point is within the range of the robot workspace,
        otherwise it returns False.
        """

        # We consider that the range of the pelican robot workspace is within
        # the area of the circle formed by the radius that forms approximately
        # the sum of its links.
        c = np.linspace(0, 2 * np.pi)
        r_space = [[(2 * 0.26) * np.cos(c[i]), (2 * L1) * np.sin(c[i])]
                    for i in range(len(c))]
        x_min = min([r_space[x][0] for x in range(len(r_space))])
        y_min = min([r_space[y][1] for y in range(len(r_space))])

        x_max = max([r_space[x][0] for x in range(len(r_space))])
        y_max = max([r_space[y][1] for y in range(len(r_space))])

        if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
            return True
        else:
            print("""The range of values is:
                x[{}, {}]
                y[{}, {}]""".format(x_min, x_max, y_min, y_max))

            return False

    @staticmethod
    def chk_kwarg(kwarg):
        """
        Function to check that kwarg is in the correct form and return the type
        of controller to use
        """
        if len(kwarg) == 0:
            kwarg['control_law'] = 'PD'
        elif len(kwarg) != 1:
            raise ValueError('kwarg attributes only one parameter')
        for key in kwarg.keys():
            if key not in {'control_law'}:
                raise ValueError('Controller has no attribute {}'.format(key))
            elif kwarg[key] not in {'PD', 'PD-GC', 'PD-dGC'}:
                raise ValueError('{} has no attribute {}'.format(key, kwarg[key]))
            else:
                return kwarg[key]

    @staticmethod
    def G(q1, q2):
        """
        Funtion to evaluate vector of gravitational forces and torque

        It is used as a function so that it can be used in the controller
        equation if necessary.
        """

        g_11 = (M1 * LC1 + M2 * L1) * GR * np.sin(q1) + M2 * LC2 * GR * np.sin(q1 + q2)
        g_21 = M2 * LC2 * GR * np.sin(q1 + q2)
        return [g_11, g_21]

    def RK4(self, ti, qi, vi, tf, h=0.001, display=False):
        """
        function to solve by Runge-Kutta 4th Order

        Parameters
        ----------
        ti : Value of the initial t
        qi : Value of the initial angles [q1, q2]
        vi : Value of the initial velocities [qp1, qp2]
        tf : time(s) that you want to evaluate in the diff system
        h : Integration step
        display : bool
            If it is True, it shows in real time how the angles of each link
            changed and the error reduction to the desired point.

        Returns
        -------
        qi : Values of final angles [q1, q2] to desired position
        vi : Values of final velocities [qp1, qp2] to desired position
        """

        if h > 0.01:
            raise ValueError('{} is greater than 0.01, set h <= 0.01'.format(h))

        print("""
              RK4 method will do {} iterations to the desired point {}
              """.format(len(np.arange(ti, tf, h)), self.dp))

        for _ in np.arange(ti, tf, h):
            qt = self.controller(qi, vi)
            self.qerr.append(qt)

            k1 = vi
            m1 = self.MCG(qi, vi)

            k2 = vi + np.dot(m1, h / 2)
            m2 = self.MCG(qi + np.dot(k1, h / 2), vi + np.dot(m1, h / 2))

            k3 = vi + np.dot(m2, h / 2)
            m3 = self.MCG(qi + np.dot(k2, h / 2), vi + np.dot(m2, h / 2))

            k4 = vi + np.dot(m3, h)
            m4 = self.MCG(qi + np.dot(k3, h), vi + np.dot(m3, h))

            qi += (h / 6) * (k1 + 2 * (k2 + k3) + k4)
            self.qs.append([qi[0], qi[1]])

            vi += (h / 6) * (m1 + 2 * (m2 + m3) + m4)
            self.vs.append([vi[0], vi[1]])

            ti += h
            self.ts.append(ti)

        if display == True:
            realtime().show(self.ts, self.qs, self.qerr, self.dp, self.ctrl_type)

        return qi, vi

    def controller(self, qst, qpst):
        """
        Types of position control:
            - Proportional control plus velocity feedback and Proportional
              Derivative (PD) control
            - PD control with gravity compensation
            - PD control with desired gravity compensation

        The controller's gains were modified to try to reduce the error

        Parameters
        ----------
        qst : Values of angles position q(t), time dependent
        qpst : Values of each link velocities, time dependent

        Return
        ------
        qt : error value of desired position
        """

        qds = inverse_k(self.dp[0], self.dp[1])
        qt = [qds[0] - qst[0], qds[1] - qst[1]]

        if self.ctrl_type == 'PD':
            self.tau = np.dot(self.kp, qt) - np.dot(self.kv, qpst)
        elif self.ctrl_type == 'PD-GC':
            self.tau = np.dot(self.kp, qt) - np.dot(self.kv, qpst) + pelican_robot.G(qst[0], qst[1])
        elif self.ctrl_type == 'PD-dGC':
            self.tau = np.dot(self.kp, qt) - np.dot(self.kv, qpst) + pelican_robot.G(qds[0], qds[1])

        return qt

    def MCG(self, q, v):
        """
        Dynamic model of pelican robot with Controller and gravity compensation
        tau = M(q)\Ddot{q} + C(q, \dot{q})\dot{q} + G(q)
            M(q) := Inertial Matrix
            C(q, \dot{q}) := Centrifugal and Coriolis Forces Matrix
            G(q) := Gravitational Torques Vector

        Parameters
        ----------
        q : Angles of each link  [q1, q2]
        v : Velocitys of each link [qp1, qp2]

        Return
        ------
        qpp : Acelerations of each link [qpp1, qpp2]
        """

        # Inertial Matrix
        m_11 = M1 * LC1 ** 2 + M2 * (L1 ** 2 + LC2 ** 2 + 2 * L1 * LC2 * np.cos(q[1])) + I1 + I2
        m_12 = M2 * (LC2 ** 2 + L1 * LC2 * np.cos(q[1])) + I2
        m_21 = M2 * (LC2 ** 2 + L1 * LC2 * np.cos(q[1])) + I2
        m_22 = M2 * LC2 ** 2 + I2

        m = [
            [m_11, m_12],
            [m_21, m_22]
        ]

        # Vector of centrifugal and Coriolis forces
        c_11 = - ((M2 * L1 * LC2 * np.sin(q[1])) * v[1])
        c_12 = - ((M2 * L1 * LC2 * np.sin(q[1])) * (v[0] + v[1]))
        c_21 =   ((M2 * L1 * LC2 * np.sin(q[1])) * v[1])
        c_22 = 0.0

        c = [
            [c_11, c_12],
            [c_21, c_22]
        ]

        if len(self.tau) == 0:
            str_error = 'The tau value of the "controller function" is needed.'
            raise ValueError(str_error)
        else:
            __tau = self.tau

        minv = np.linalg.inv(m)
        cqp = np.dot(c, v)
        qpp = np.matmul(minv, (__tau - cqp - pelican_robot.G(q[0], q[1])))

        return qpp

    def plot_velocity_bhvr(self):
        """
        Function to graph the velocity behavior of each iteration
        """

        if len(self.vs) == 0:
            raise ValueError("""First run RK4 to obtain each iteration of the
                solution for the desired position to be able to graph""")

        # Change icon window
        if os.path.exists('../pelcnrbt/images'):
            plc_anim_w = plt.get_current_fig_manager()
            img = PhotoImage(file='images/pelican-robot-icon.png')
            plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

        plt.title('Graph of velocity Behavior')
        plt.plot(self.ts, [(self.vs[i][0]) for i in range(len(self.vs))], "r--",
                 label = "$ \\dot{q_1} $")
        plt.plot(self.ts, [(self.vs[i][1]) for i in range(len(self.vs))], "b--",
                 label = "$ \\dot{q_2} $")
        plt.legend()
        plt.grid()
        plt.xlabel("$ t(s) $", fontsize='large')
        plt.ylabel("$ \\frac{rad}{s} $", rotation='horizontal', fontsize='x-large')

    def plot_q_error(self):
        """
        Function to graph the behavior of how the error of each link was reduced
        until reaching the desired position.
        """

        if len(self.qerr) == 0:
            raise ValueError("""First run RK4 to obtain each iteration of the
                solution for the desired position to be able to graph""")

        # Change icon window
        if os.path.exists('../pelcnrbt/images'):
            plc_anim_w = plt.get_current_fig_manager()
            img = PhotoImage(file='images/pelican-robot-icon.png')
            plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

        plt.title("Graph of $ \\tilde{q} $")
        # Plot
        plt.plot(self.ts, [(self.qerr[i][0]) for i in range(len(self.qerr))],
                 "r--", label = "$ \\tilde{q_1} $")
        plt.plot(self.ts, [(self.qerr[i][1]) for i in range(len(self.qerr))],
                 "b--", label = "$ \\tilde{q_2} $")
        plt.legend()
        plt.grid()
        plt.xlabel("$ t(s) $", fontsize='large')
        plt.ylabel("$ \\tilde{q}(rad) $", rotation='horizontal', fontsize='large')

    def plot_trajectory(self, stp):
        """
        Function to graph trajectory of desired position

        Parameters
        ----------
        stp : int number of plot steps in range (1 - 100)
        """

        if len(self.qs) == 0:
            raise ValueError("""First run RK4 to obtain each iteration of the
                solution for the desired position to be able to graph""")

        if not 1 <= stp <= 100:
            raise ValueError('The number should be in range (1 - 100)')

        fig, ax = plt.subplots()
        # Change icon window
        if os.path.exists('../pelcnrbt/images'):
            plc_anim_w = plt.get_current_fig_manager()
            img = PhotoImage(file='images/pelican-robot-icon.png')
            plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

        # Work space of robot
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)

        for n in np.arange(0, len(self.qs), len(self.qs) // stp):
            # Draw each number of steps
            plt.title('Pelican Robot: Desired Position trajectory with Position Control')
            plot_link([0.0, 0.0], direct_k(self.qs[n][0], self.qs[n][1])[0], '#83EB94')
            plot_link(direct_k(self.qs[n][0], self.qs[n][1])[0],
                      direct_k(self.qs[n][0], self.qs[n][1])[1], '#83EB94')
            ax.grid('on')
            ax.set_aspect('equal', 'box')

        # Draw point in desired position
        ax.scatter(self.dp[0], self.dp[1], marker='X', s=100, facecolor='#65009C')
        # Draw initial position
        plot_link([0.0, 0.0], direct_k(self.qs[0][0], self.qs[0][1])[0], 'k--')
        plot_link(direct_k(self.qs[0][0], self.qs[0][1])[0],
                  direct_k(self.qs[0][0], self.qs[0][1])[1], 'k--')
        # Draw Final Position
        plot_link([0.0, 0.0], direct_k(self.qs[len(self.qs) - 1][0], self.qs[len(self.qs) - 1][1])[0],
                  'r', label='$L_1$')
        plot_link(direct_k(self.qs[len(self.qs) - 1][0], self.qs[len(self.qs) - 1][1])[0],
                  direct_k(self.qs[len(self.qs) - 1][0], self.qs[len(self.qs) - 1][1])[1],
                  'b', label='$L_2$')
        ax.legend()

    def values(self):
        """
        Function to return values of each iteration in RK4

        Returns
        -------
        us : Angle values of each iteration to desired position
        vs : Velocities values of each iteration to desired position
        ts : Time values for graph
        """

        return self.qs, self.vs, self.ts

if __name__ == '__main__':
    # Desired position
    dp = [0.26, 0.13]
    # Gains
    kp = [[30.0, 0.0],
          [0.0, 30.0]]
    kv = [[7.0, 0.0],
          [0.0, 3.0]]
    # Initial values of angles and velocities
    qi = [0.0, 0.0]
    vi = [0.0, 0.0]
    ti = 0.0
    tf = 1.0

    sim = pelican_robot(dp, kp, kv, control_law='PD-GC')
    qsf, qpsf = sim.RK4(ti, qi, vi, tf, display=True)

    print('==================================================================')
    print('Angles for desired position: [{}, {}]'.format(dp[0], dp[1]))
    print('q1 = {} rad, q2 = {} rad.'.format(qsf[0], qsf[1]))

    print('Close window of error graph...')
    sim.plot_q_error()
    plt.show()

    print('Close window of trajectory plot...')
    sim.plot_trajectory(50)
    plt.show()
