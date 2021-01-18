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
from matplotlib.animation import FuncAnimation

def plot_link(p_i, p_f, *args, **kwargs):
    """
    Function to graph a link in R²

    Parameters
    -----------
    pi : Initial point of link in R²
    pf : Final point of link in R²
    """

    plt.plot([p_i[0], p_i[0] + p_f[0]], [p_i[1], p_i[1] + p_f[1]], *args, **kwargs)
    # If the color is not specified, the np.scatter decorator will default to itself.
    if len(args) == 0:
        plt.scatter(p_i[0], p_i[1])
    # Otherwise the color is used regardless of the line style. Since the line
    # style does not attribute to np.sactter.
    else:
        color = ''.join([args[0][i] for i in range(len(args[0]))
                         if args[0][i] not in list(plt.Line2D.lineStyles)])
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

    l1 = l2 = 0.26 # In meters
    link_1 = [l1 * np.sin(q1), - l1 * np.cos(q1)]
    link_2 = [l2 * np.sin(q1 + q2), - l2 * np.cos(q1 + q2)]
    fnl_elmnt = [link_1[0] + link_2[0], link_1[1] + link_2[1]]

    return link_1, link_2, fnl_elmnt

def inverse_k(Px, Py):
    """
    Inverse kinematic of pelican robot:
        Obtain angles for each link to desired position

    Parameters
    ----------
    Px : Value of x-position desired
    Py : Value of y-position desired

    Returns
    -------
    q1 : Angle in radians of link 1 for the desired position
    q2 : Angle in radians of link 2 for the desired position
    """

    l1 = l2 = 0.26 # In meters

    K = ((Px ** 2 + Py ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
    q2 = np.arctan2(np.sqrt(1 - (K ** 2)), K)

    __tmp = np.arctan2((l2 * np.sin(q2)), (l1 + l2 * np.cos(q2)))
    q1 = np.arctan2(Px, - Py) - __tmp

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
        self.fig.canvas.set_window_title('Pelican Robot: Simulation')
        # Change icon window
        if os.path.exists('../images'):
            plc_anim_w = plt.get_current_fig_manager()
            img = PhotoImage(file='images/pelican-robot-icon.png')
            plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

        # Limits and appearance of the pelican robot animation
        self.robot.set_xlim((-0.6, 0.6))
        self.robot.set_ylim((-0.6, 0.6))
        self.robot.set_aspect('equal', adjustable='box')
        self.robot.grid('on')

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

        # Get limits of qt plot
        self.qt.set_xlim((min(ts), max(ts)))
        qtmin = min(min([qt[i][0] for i in range(len(qt))]), min([qt[i][1] for i in range(len(qt))]))
        qtmax = max(max([qt[i][0] for i in range(len(qt))]), max([qt[i][1] for i in range(len(qt))]))
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
        q1e = [qt[n][0] for n in range(len(qt))]
        q2e = [qt[n][1] for n in range(len(qt))]

        def animate(i):
            """
            Generate animation
            """

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
    Simulate a desired potion for the pelican robot by means of a PD controller
    with gravity compensation and graph the behaviors

    ...

    Attributes
    ----------
    dp : list
        Desired position [Px, Py]
    kp : list array
        Position gain in matrix form.
    kv : list array
        Velocity gain in matrix form
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
        self.error = []
        self.tau = [0.0, 0.0]

    @staticmethod
    def pts_in_range(point):
        """
        Returns True if the point is within the range of the robot workspace,
        otherwise it returns False.
        """

        # We consider that the range of the robot's workspace is within the area
        # of the circle formed by the radius that forms approximately the sum of its links.
        c = np.linspace(0, 2 * np.pi)
        r_space = [[(2 * 0.25) * np.cos(c[i]), (2 * 0.25) * np.sin(c[i])]
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
            elif kwarg[key] == 'PD':
                return kwarg[key]
            elif kwarg[key] == 'PD-GC':
                return kwarg[key]
            elif kwarg[key] == 'PD-dGC':
                return kwarg[key]

    @staticmethod
    def G(q1, q2):
        """
        Funtion to evaluate vector of gravitational forces and torque

        It is used as a function so that it can be used in the controller
        equation if necessary.
        """

        # Required robot parameters
        l1 = l2 = 0.26 # meters
        lc1 = 0.0983 # meters
        lc2 = 0.0229 # meters
        m1 = 6.5225 # Kg
        m2 = 2.0458 # Kg
        g = 9.81 # m / s²

        G11 = (m1 * lc1 + m2 * l1) * g * np.sin(q1) + m2 * lc2 * g * np.sin(q1 + q2)
        G12 = m2 * lc2 * g * np.sin(q1 + q2)
        return [G11, G12]

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

        st = np.arange(ti, tf, h)

        for _ in st:
            qt = self.controller(qi, vi)
            self.error.append(qt)

            K1 = vi
            m1 = self.BCG(qi, vi)

            K2 = vi + np.dot(m1, h / 2)
            m2 = self.BCG(qi + np.dot(K1, h / 2), vi + np.dot(m1, h / 2))

            K3 = vi + np.dot(m2, h / 2)
            m3 = self.BCG(qi + np.dot(K2, h / 2), vi + np.dot(m2, h / 2))

            K4 = vi + np.dot(m3, h)
            m4 = self.BCG(qi + np.dot(K3, h), vi + np.dot(m3, h))

            qi += (h / 6) * (K1 + 2 * (K2 + K3) + K4)
            self.qs.append([qi[0], qi[1]])

            vi += (h / 6) * (m1 + 2 * (m2 + m3) + m4)
            self.vs.append([vi[0], vi[1]])

            ti += h
            self.ts.append(ti)

        if display == True:
            realtime().show(self.ts, self.qs, self.error, self.dp, self.ctrl_type)

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

    def BCG(self, q, v):
        """
        Dynamic model of pelican robot with Controller and gravity compensation

        Parameters
        ----------
        q : Angles of each link  [q1, q2]
        v : Velocitys of each link [qp1, qp2]

        Return
        ------
        qpp : Acelerations of each link [qpp1, qpp2]
        """

        # Parameters of the robot
        l1 = l2 = 0.26 # meters
        lc1 = 0.0983 # meters
        lc2 = 0.0229 # meters
        m1 = 6.5225 # Kg
        m2 = 2.0458 # Kg
        I1 = 0.1213 # Kg m²
        I2 = 0.01616 # Kg m²
        g = 9.81 # m / s²

        # Inertial Matrix
        B11 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(q[1])) + I1 + I2
        B12 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(q[1])) + I2
        B21 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(q[1])) + I2
        B22 = m2 * lc2 ** 2 + I2

        B = [
            [B11, B12],
            [B21, B22]
        ]

        # Vector of centrifugal and Coriolis forces
        C11 = - ((m2 * l1 * lc2 * np.sin(q[1])) * v[1])
        C12 = - ((m2 * l1 * lc2 * np.sin(q[1])) * (v[0] + v[1]))
        C21 =   ((m2 * l1 * lc2 * np.sin(q[1])) * v[1])
        C22 = 0.0

        C = [
            [C11, C12],
            [C21, C22]
        ]

        if len(self.tau) == 0:
            str_error = 'The tau value of the "controller function" is needed.'
            raise ValueError(str_error)
        else:
            __tau = self.tau

        Bi = np.linalg.inv(B)
        Cqp = np.dot(C, v)
        qpp = np.matmul(Bi, (__tau - Cqp - pelican_robot.G(q[0], q[1])))

        return qpp

    def plot_velocity_bhvr(self):
        """
        Function to graph the velocity behavior of each iteration
        """

        if len(self.vs) == 0:
            raise ValueError("""First run RK4 to obtain each iteration of the
                solution for the desired position to be able to graph""")

        # Change icon window
        if os.path.exists('../images'):
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
        plt.xlabel("$ t $", fontsize='large')
        plt.ylabel("$ \\frac{rad}{s} $", rotation='horizontal', fontsize='x-large')

    def plot_q_error(self):
        """
        Function to graph the behavior of how the error of each link was reduced
        until reaching the desired position.
        """

        if len(self.error) == 0:
            raise ValueError("""First run RK4 to obtain each iteration of the
                solution for the desired position to be able to graph""")

        # Change icon window
        if os.path.exists('../images'):
            plc_anim_w = plt.get_current_fig_manager()
            img = PhotoImage(file='images/pelican-robot-icon.png')
            plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

        plt.title("Graph of $ \\tilde{q} $")
        # Plot
        plt.plot(self.ts, [(self.error[i][0]) for i in range(len(self.error))],
                 "r--", label = "$ \\tilde{q_1} $")
        plt.plot(self.ts, [(self.error[i][1]) for i in range(len(self.error))],
                 "b--", label = "$ \\tilde{q_2} $")
        plt.legend()
        plt.grid()
        plt.xlabel("$ t $", fontsize='large')
        plt.ylabel("$ \\tilde{q} $", rotation='horizontal', fontsize='large')

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
        if os.path.exists('../images'):
            plc_anim_w = plt.get_current_fig_manager()
            img = PhotoImage(file='images/pelican-robot-icon.png')
            plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

        # Work space of robot
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)

        _stp_ = np.arange(0, len(self.qs), len(self.qs) // stp)

        for n in _stp_:
            # Draw each number of steps
            plt.title('Pelican Robot: Desired Position trajectory with Position Control')
            plot_link([0.0, 0.0], direct_k(self.qs[n][0], self.qs[n][1])[0], '#83EB94')
            plot_link(direct_k(self.qs[n][0], self.qs[n][1])[0],
                      direct_k(self.qs[n][0], self.qs[n][1])[1], '#83EB94')
            ax.grid('on')

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
    qsf, qpsf = sim.RK4(ti, qi, vi, tf)

    print('==================================================================')
    print('Angles for desired position: [{}, {}]'.format(dp[0], dp[1]))
    print('q1 = {} rad, q2 = {} rad.'.format(qsf[0], qsf[1]))

    print('Close window of error graph...')
    sim.plot_q_error()
    plt.show()

    print('Close window of trajectory plot...')
    sim.plot_trajectory(50)
    plt.show()
