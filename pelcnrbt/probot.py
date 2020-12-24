# -*- coding: utf-8 -*-

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
    plt.scatter(p_i[0], p_i[1], facecolor=args[0])

def direct_k(q1, q2):
    """
    Function to obtain x-y-cordinates of each link

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

class realtime:
    def __init__(self):
        self.fig, (self.robot, self.qt) = plt.subplots(2, 1)
        self.fig.canvas.set_window_title('Pelican Robot: Simulation')
        # Change icon window
        plc_anim_w = plt.get_current_fig_manager()
        img = PhotoImage(file='images/pelican-robot-icon.png')
        plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

        self.robot.set_xlim((-0.6, 0.6))
        self.robot.set_ylim((-0.6, 0.6))
        self.robot.set_aspect('equal', adjustable='box')
        self.robot.grid('on')

        self.qt.set_title("Graph of how the error ($ \\tilde{q} $) was reduced.")
        self.qt.set_xlabel("$ t(s) $", fontsize='large')
        self.qt.set_ylabel("$ \\tilde{q} $", rotation='horizontal', fontsize='large')
        self.qt.grid('on', linestyle='--')

    def show(self, ts, qs, qt, dp):
        self.qt.set_xlim((min(ts), max(ts)))
        qtmin = min(min([qt[i][0] for i in range(len(qt))]), min([qt[i][1] for i in range(len(qt))]))
        qtmax = max(max([qt[i][0] for i in range(len(qt))]), max([qt[i][1] for i in range(len(qt))]))
        self.qt.set_ylim((qtmin, qtmax))
        str_controller = r'$  \tau = K_{p} \tilde{q} - K_{v} \dot{q} + g(q) $'
        str_qt = r'$ \tilde{q} := q_d - q(t) $'
        self.qt.text((max(ts) / 2), (qtmax), 'Controller equation',
                     horizontalalignment='center', verticalalignment='top')
        self.qt.text((max(ts) / 2), (qtmax - 200 * 0.001), str_controller,
                     horizontalalignment='center', verticalalignment='top')
        self.qt.text((max(ts) / 2), (qtmax - 500 * 0.001), str_qt, horizontalalignment='center',
                     verticalalignment='top')

        self.robot.set_title(r"Pelican Robot: Trajectory to [{}, {}]".format(dp[0], dp[1]))
        self.robot.scatter(dp[0], dp[1], facecolor='k')
        link_1, = self.robot.plot([], [], 'r', lw=2)
        link_2, = self.robot.plot([], [], 'b', lw=2)

        q1_error, = self.qt.plot([], [], 'r', lw=2, label="$ \\tilde{q_{1}} $")
        q2_error, = self.qt.plot([], [], 'b', lw=2, label="$ \\tilde{q_{2}} $")
        self.qt.legend(loc='upper right')
        q1e = [qt[n][0] for n in range(len(qt))]
        q2e = [qt[n][1] for n in range(len(qt))]

        def animate(i):
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

    Methods
    -------
    RK4(ti, ui, vi, tf, h):
        Runge-Kutta 4th order to solve system ODE's to obtain angles and velocities.

    inverse(Px, Py):
        Obtain the inverse kinematic of pelican robot.

    controller(qs, qps):
        - Controller that returns the error of each angle of the desired position.
        - Generate the controller tau with the set gains to use within the class.

    BCG(v, u):
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

    def __init__(self, dp, kp, kv):
        """
        Constructor

        Parameters
        ----------
        dp : Desired position ([Px, Py] in R²)
        kp : Position gain. e.g. [[30.0, 0.0],[0.0, 30.0]]
        kv : Velocity gain. e.g. [[7.0, 0.0],[0.0, 2.0]]
        """

        self.dp = dp
        self.kp = kp
        self.kv = kv
        self.ts = []
        self.qs = []
        self.vs = []
        self.error = []
        self.tau = 0

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
            If it is true, it shows in real time how the angles of each link
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
            realtime().show(self.ts, self.qs, self.error, self.dp)

        return qi, vi

    def inverse(self, Px, Py):
        """
        Function to obtain the inverse kinematic of pelican robot

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

        a = np.arctan2(Px, -Py)
        b = np.arctan2((l2 * np.sin(q2)), (l1 + l2 * np.cos(q2)))
        q1 = a - b

        return q1, q2

    def controller(self, qst, qpst):
        """
        Proportional Control plus Velocity Feedback and PD Control

        The controller's gains were modified to try to reduce the error

        Parameters
        ----------
        qst : Values of angles position q(t), time dependent
        qpst : Values of each link velocities, time dependent

        Return
        ------
        qt : error value of desired position
        """

        qds = self.inverse(self.dp[0], self.dp[1])
        qt = [qds[0] - qst[0], qds[1] - qst[1]]

        self.tau = np.dot(self.kp, qt) - np.dot(self.kv, qpst)

        return qt

    def BCG(self, q, v):
        """
        Dynamic model of pelican robot with Controller and gravity compensation

        Parameters
        ----------
        v : Velocitys of each link [qp1, qp2]
        u : Angles of each link  [q1, q2]

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

        # Vector of gravitational forces and torques
        G11 = (m1 * lc1 + m2 * l1) * g * np.sin(q[0]) + m2 * lc2 * g * np.sin(q[0] + q[1])
        G12 = m2 * lc2 * g * np.sin(q[0] + q[1])
        G = [G11, G12]

        if len(self.tau) == 0:
            str_error = 'The tau value of the "controller function" is needed.'
            raise ValueError(str_error)
        else:
            # Gravity Compensation
            __tau = self.tau + G

        Bi = np.linalg.inv(B)
        Cqp = np.dot(C, v)
        qpp = np.matmul(Bi, (__tau - Cqp - G))

        return qpp

    def plot_velocity_bhvr(self):
        """
        Function to graph the velocity behavior of each iteration
        """

        if len(self.vs) == 0:
            str_error = 'First run RK4 to obtain each iteration of the solution for the desired position to be able to graph'
            raise ValueError(str_error)

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
            str_error = 'First run RK4 to obtain each iteration of the solution for the desired position to be able to graph'
            raise ValueError(str_error)

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
            str_error = 'First run RK4 to obtain each iteration of the solution for the desired position to be able to graph'
            raise ValueError(str_error)

        if not 1 <= stp <= 100:
            raise ValueError('The number should be in range (1 - 100)')

        fig, ax = plt.subplots()
        # Work space of robot
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)

        _stp_ = np.arange(0, len(self.qs), len(self.qs) // stp)

        for n in _stp_:
            # Draw each number of steps
            plt.title('Pelican Robot: Desired Position trajectory with PD-Controller')
            plot_link([0.0, 0.0], direct_k(self.qs[n][0], self.qs[n][1])[0], '#83EB94')
            plot_link(direct_k(self.qs[n][0], self.qs[n][1])[0],
                      direct_k(self.qs[n][0], self.qs[n][1])[1], '#83EB94')
            ax.grid('on')

        # Draw point in desired position
        ax.scatter(self.dp[0], self.dp[1], marker='X', s=100, facecolor='#65009C')
        # Draw Home Position
        ax.plot([0.0, 0.0], [0.0, -0.52], '--k')
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
    pd = [0.26, 0.13]
    # Gains
    kp = [[30.0, 0.0],
          [0.0, 30.0]]
    kv = [[7.0, 0.0],
          [0.0, 3.0]]
    # Initial values
    qi = [0.0, 0.0]
    vi = [0.0, 0.0]
    ti = 0.0
    tf = 1.0

    sim = pelican_robot(pd, kp, kv)
    qsf, qppsf = sim.RK4(ti, qi, vi, tf)

    print('==================================================================')
    print('Angles for desired position: [{}, {}]'.format(pd[0], pd[1]))
    print('q1 = {} rad, q2 = {} rad.'.format(qsf[0], qsf[1]))

    print('Close window of error graph...')
    sim.plot_q_error()
    plt.show()

    print('Close window of trajectory plot...')
    sim.plot_trajectory(50)
    plt.show()
