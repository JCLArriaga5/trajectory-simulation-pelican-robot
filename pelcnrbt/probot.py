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
from utils import *

OS = sys.platform
# Integration step
H = 0.001

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
    def run(ti, qi, vi, tf, display=False):
        Function to run simulation using runge kutta 4th order method to solve
        the trajectory.

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

    empty_values():
        Funtion to set empty values.
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
        self.ctrl_type = self.__chk_kwarg(kwarg)
        self.empty_values()

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

    def __chk_kwarg(self, kwarg):
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

    def run(self, ti, qi, vi, tf, display=False):
        """
        Function to run simulation using runge kutta 4th order method to solve
        the trajectory.

        Parameters
        ----------
        ti : Value of the initial t
        qi : Value of the initial angles [q1, q2]
        vi : Value of the initial velocities [qp1, qp2]
        tf : time(s) that you want to evaluate in the diff system
        display : bool
            If it is True, it shows in real time how the angles of each link
            changed and the error reduction to the desired point.

        Returns
        -------
        qf : Values of final angles [q1, q2] to desired position
        vf : Values of final velocities [qp1, qp2] to desired position
        """

        self.empty_values()

        print("""
              RK4 method will do {} iterations to the desired point {}
              """.format(len(np.arange(ti, tf, H)), self.dp))

        qf, vf = self.__rk4(ti, qi, vi, tf)

        if display == True:
            message_dialog('Information', 'When the simulation finishes, close the window to continue with the script.')
            realtime().show(self.ts, self.qs, self.qerr, self.dp, self.ctrl_type)

        return qf, vf

    def __rk4(self, ti, qi, vi, tf):
        """
        Runge-Kutta 4th Order method.

        Parameters
        ----------
        ti : Value of the initial t
        qi : Value of the initial angles [q1, q2]
        vi : Value of the initial velocities [qp1, qp2]
        tf : time(s) that you want to evaluate in the diff system

        Returns
        -------
        qi : Values of final angles [q1, q2] to desired position
        vi : Values of final velocities [qp1, qp2] to desired position
        """

        for _ in np.arange(ti, tf, H):
            qt = self.__controller(qi, vi)
            self.qerr.append(qt)

            k1 = vi
            m1 = self.__mcg(qi, vi)

            k2 = vi + np.dot(m1, H / 2)
            m2 = self.__mcg(qi + np.dot(k1, H / 2), vi + np.dot(m1, H / 2))

            k3 = vi + np.dot(m2, H / 2)
            m3 = self.__mcg(qi + np.dot(k2, H / 2), vi + np.dot(m2, H / 2))

            k4 = vi + np.dot(m3, H)
            m4 = self.__mcg(qi + np.dot(k3, H), vi + np.dot(m3, H))

            qi += (H / 6) * (k1 + 2 * (k2 + k3) + k4)
            self.qs.append([qi[0], qi[1]])

            vi += (H / 6) * (m1 + 2 * (m2 + m3) + m4)
            self.vs.append([vi[0], vi[1]])

            ti += H
            self.ts.append(ti)

        return qi, vi

    def __controller(self, qst, qpst):
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
            self.tau = np.dot(self.kp, qt) - np.dot(self.kv, qpst) + Dynamic_model.G(qst[0], qst[1])
        elif self.ctrl_type == 'PD-dGC':
            self.tau = np.dot(self.kp, qt) - np.dot(self.kv, qpst) + Dynamic_model.G(qds[0], qds[1])

        return qt

    def __mcg(self, q, v):
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

        if len(self.tau) == 0:
            str_error = 'The tau value of the "controller function" is needed.'
            raise ValueError(str_error)
        else:
            __tau = self.tau

        minv = np.linalg.inv(Dynamic_model.M(q[1]))
        cqp = np.dot(Dynamic_model.C(q[1], v[0], v[1]), v)
        qpp = np.matmul(minv, (__tau - cqp - Dynamic_model.G(q[0], q[1])))

        return qpp

    def plot_velocity_bhvr(self):
        """
        Function to graph the velocity behavior of each iteration
        """

        if len(self.vs) == 0:
            raise ValueError("""First run RK4 to obtain each iteration of the
                solution for the desired position to be able to graph""")

        # Change icon window
        set_icon_window('pelican-robot-icon.png')
        # Change Figure title
        set_title('Graph')

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
        set_icon_window('pelican-robot-icon.png')
        # Change Figure title
        set_title('Graph')

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
        set_icon_window('pelican-robot-icon.png')
        # Change Figure title
        set_title('Graph')
        # Work space of robot
        ax.set_xlim((-0.6, 0.6))
        ax.set_ylim((-0.6, 0.6))
        ax.set_xticks(np.arange(-0.6, 0.6, 0.1))
        ax.set_yticks(np.arange(-0.6, 0.6, 0.1))
        ax.grid(which='both')

        for n in np.arange(0, len(self.qs), len(self.qs) // stp):
            # Draw each number of steps
            plt.title('Pelican Robot: Desired Position trajectory with Position Control')
            plot_link([0.0, 0.0], direct_k(self.qs[n][0], self.qs[n][1])[0], '#83EB94')
            plot_link(direct_k(self.qs[n][0], self.qs[n][1])[0],
                      direct_k(self.qs[n][0], self.qs[n][1])[1], '#83EB94')
            ax.grid('on')
            ax.set_aspect('equal', adjustable='box')

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

    def empty_values(self):
        """
        Funtion to set empty values
        """

        self.ts = []
        self.qs = []
        self.vs = []
        self.qerr = []
        self.tau = [0.0, 0.0]

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
    qsf, qpsf = sim.run(ti, qi, vi, tf, display=True)

    print('==================================================================')
    print('Angles for desired position: [{}, {}]'.format(dp[0], dp[1]))
    print('q1 = {} rad, q2 = {} rad.'.format(qsf[0], qsf[1]))

    print('Close window of error graph...')
    sim.plot_q_error()
    message_dialog('Information', 'When you finish viewing the error graph, close the window.')
    plt.show()

    print('Close window of trajectory plot...')
    sim.plot_trajectory(50)
    message_dialog('Information', 'When you finish viewing the trajectory plot, close the window.')
    plt.show()
