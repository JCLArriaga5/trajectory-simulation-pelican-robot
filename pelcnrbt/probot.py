# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_link(p_i, p_f, *args, **kwargs):
    """
    Function to graph a link in R²

    Parameters
    -----------
    pi : Initial point of link in R²
    pf : Final point of link in R²

    Example
    -------
    plt.plot([0, 0], [5, 5])
    """
    plt.plot([p_i[0], p_i[0] + p_f[0]], [p_i[1], p_i[1] + p_f[1]], *args, **kwargs)
    plt.scatter(p_i[0], p_i[1], facecolor=args[0])

def plcn_drct_kinematic(q1, q2):
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
    l1 = l2 = 0.26
    link_1 = [l1 * np.sin(q1), - l1 * np.cos(q1)]
    link_2 = [l2 * np.sin(q1 + q2), - l2 * np.cos(q1 + q2)]
    fnl_elmnt = [link_1[0] + link_2[0], link_1[1] + link_2[1]]

    return link_1, link_2, fnl_elmnt


class pelican_robot:
    """
    Class to simulate a desired potion for the pelican robot by means of a PD
    controller with gravity compensation and graph the behaviors.
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
        self.us = []
        self.vs = []
        self.error = []
        self.tau = 0

    def RK4(self, ti, ui, vi, tf, h):
        """
        function to solve by Runge-Kutta 4th Order

        Parameters
        ----------
        ti : Value of the initial t
        ui : Value of the initial angles [q1, q2]
        vi : Value of the initial velocities [qp1, qp2]
        tf : time(s) that you want to evaluate in the diff system
        h : Integration step

        Returns
        -------
        ui : Values of final angles [q1, q2] to desired position
        vi : Values of final velocities [qp1, qp2] to desired position
        """
        st = np.arange(ti, tf, h)

        for _ in st:
            qt = self.controller(ui, vi)
            self.error.append(qt)

            K1 = self.f(vi)
            m1 = self.BCG(vi, ui)

            K2 = self.f(vi + np.dot(m1, h / 2))
            m2 = self.BCG(vi + np.dot(m1, h / 2), ui + np.dot(K1, h / 2))

            K3 = self.f(vi + np.dot(m2, h / 2))
            m3 = self.BCG(vi + np.dot(m2, h / 2), ui + np.dot(K2, h / 2))

            K4 = self.f(vi + np.dot(m3, h))
            m4 = self.BCG(vi + np.dot(m3, h), ui + np.dot(K3, h))

            ui += (h / 6) * (K1 + 2 * K2 + 2 * K3 + K4)
            self.us.append([ui[0], ui[1]])

            vi += (h / 6) * (m1 + 2 * m2 + 2 * m3 + m4)
            self.vs.append([vi[0], vi[1]])

            ti += h
            self.ts.append(ti)

        # print('The number of iteratios was {}'.format(len(self.us)))
        return ui, vi

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
        l1 = 0.26
        l2 = 0.26

        K = ((Px ** 2 + Py ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
        q2 = np.arctan2(np.sqrt(1 - (K ** 2)), K)

        a = np.arctan2(Px, -Py)
        b = np.arctan2((l2 * np.sin(q2)), (l1 + l2 * np.cos(q2)))
        q1 = a - b

        return q1, q2

    def controller(self, qs, qps):
        """
        Proportional Control plus Velocity Feedback and PD Control

        The controller's gains were modified to try to reduce the error

        Parameters
        ----------
        qs : Values q(t) time dependent
        qps : Values of $\dot{q}(t)$ time dependent

        Return
        ------
        qt : error value of desired position
        """
        qds = self.inverse(self.dp[0], self.dp[1])
        qt = [qds[0] - qs[0], qds[1] - qs[1]]

        self.tau = np.dot(self.kp, qt) - np.dot(self.kv, qps)

        return qt

    def BCG(self, v, u):
        """
        Dynamic model of pelican robot with Controller

        Parameters
        ----------
        v : Velocitys of each link [qp1, qp2]
        u : Angles of each link  [q1, q2]

        Return
        ------
        qpp : Acelerations of each link [qpp1, qpp2]
        """
        # Variables of positions and speeds
        q1 = u[0]
        q2 = u[1]
        q1p = v[0]
        q2p = v[1]

        # Parameters of the robot
        l1 = 0.26
        l2 = 0.26
        lc1 = 0.0983
        lc2 = 0.0229
        m1 = 6.5225
        m2 = 2.0458
        I1 = 0.1213
        I2 = 0.01616
        g = 9.81

        # Inertial Matrix
        B11 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(q2)) + I1 + I2
        B12 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(q2)) + I2
        B21 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(q2)) + I2
        B22 = m2 * lc2 ** 2 + I2

        B = [
            [B11, B12],
            [B21, B22]
        ]

        # Vector of centrifugal and Coriolis forces
        C11 = - ((m2 * l1 * lc2 * np.sin(q2)) * q2p)
        C12 = - ((m2 * l1 * lc2 * np.sin(q2)) * (q1p + q2p))
        C21 =   ((m2 * l1 * lc2 * np.sin(q2)) * q1p)
        C22 = 0.0

        C = [
            [C11, C12],
            [C21, C22]
        ]

        # Vector of gravitational forces and torques
        G11 = (m1 * lc1 + m2 * l1) * g * np.sin(q1) + m2 * lc2 * g * np.sin(q1 + q2)
        G12 = m2 * lc2 * g * np.sin(q1 + q2)
        G = [G11, G12]

        # Gravity Compensation
        tau_ = self.tau + G

        Bi = np.linalg.inv(B)
        Cqp = np.dot(C, v)
        qpp = np.matmul(Bi, (tau_ - Cqp - G))

        return qpp

    def f(self, v):
        """
        Function necessary for computed RK4
        """
        return v

    def plot_velocity_bhvr(self):
        """
        Function to graph the velocity behavior of each iteration
        """
        if len(self.vs) == 0:
            str_error = 'First run RK4 to obtain each iteration of the solution for the desired position to be able to graph'
            raise ValueError(str_error)

        qp1_bhv = []
        qp2_bhv = []

        for _qp_ in range(len(self.vs)):
            qp1_bhv.append(self.vs[_qp_][0])
            qp2_bhv.append(self.vs[_qp_][1])

        plt.title('Graph of velocity Behavior')
        plt.plot(self.ts, qp1_bhv, "r--", label = "$ \\dot{q_1} $")
        plt.plot(self.ts, qp2_bhv, "b--", label = "$ \\dot{q_2} $")
        plt.legend()
        plt.grid()
        plt.xlabel("$ t $", fontsize='large')
        plt.ylabel("$ \\frac{rad}{s} $", rotation='horizontal', fontsize='x-large')

    def plot_q_error(self):
        """
        Function to graph the behavior of the error of each link
        """
        if len(self.error) == 0:
            str_error = 'First run RK4 to obtain each iteration of the solution for the desired position to be able to graph'
            raise ValueError(str_error)

        q1e = []
        q2e = []

        for e in range(len(self.error)):
            q1e.append(self.error[e][0])
            q2e.append(self.error[e][1])

        plt.title("Graph of $ \\tilde{q} $")
        plt.plot(self.ts, q1e, "r--", label = "$ \\tilde{q_1} $")
        plt.plot(self.ts, q2e, "b--", label = "$ \\tilde{q_2} $")
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
        if len(self.us) == 0:
            str_error = 'First run RK4 to obtain each iteration of the solution for the desired position to be able to graph'
            raise ValueError(str_error)

        if not 1 <= stp <= 100:
            raise ValueError('The number should be in range (1 - 100)')

        fig, ax = plt.subplots()
        # Work space of robot
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)

        vals = len(self.us) // stp

        _stp_ = np.arange(0, len(self.us), vals)

        for qs in _stp_:
            # Draw each number of steps
            plt.title('Pelican Robot: Desired Position trajectory with PD-Controller')
            plot_link([0.0, 0.0], plcn_drct_kinematic(self.us[qs][0], self.us[qs][1])[0], '#83EB94')
            plot_link(plcn_drct_kinematic(self.us[qs][0], self.us[qs][1])[0],
                      plcn_drct_kinematic(self.us[qs][0], self.us[qs][1])[1], '#83EB94')
            ax.grid('on')

        # Draw point in desired position
        ax.scatter(self.dp[0], self.dp[1], marker='X', s=100, facecolor='#65009C')
        # Draw Home Position
        ax.plot([0.0, 0.0], [0.0, -0.52], '--k')
        # Draw Final Position
        plot_link([0.0, 0.0], plcn_drct_kinematic(self.us[len(self.us) - 1][0], self.us[len(self.us) - 1][1])[0],
                  'r', label='$L_1$')
        plot_link(plcn_drct_kinematic(self.us[len(self.us) - 1][0], self.us[len(self.us) - 1][1])[0],
                  plcn_drct_kinematic(self.us[len(self.us) - 1][0], self.us[len(self.us) - 1][1])[1],
                  'b', label='$L_2$')
        ax.legend()

    def values(self):
        """
        Function to return values of each iteration
        """
        return self.us, self.vs, self.ts
