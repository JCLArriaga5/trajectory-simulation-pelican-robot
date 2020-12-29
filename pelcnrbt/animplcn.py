# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tkinter import PhotoImage
import argparse
from probot import *

def euclidian_d(p1, p2):
    """
    Euclidean distance between two points in R²

    Parameters
    ----------
    p1 (list): Coordinates componets of point-1 [a_1, a_2]
    p2 (list): Coordinates componets of point-1 [b_1, b_2]

    Return
    ------
    d = Euclidean distance

    Example
    -------
    >>> p1 = [2, 5]
    >>> p2 = [1, 4]
    >>> euclidian_d(p1, p2)
    1.414213

    """

    if len(p1) != 2 or len(p2) != 2:
        raise ValueError('Not in the format [a_1, a_2]')

    d = np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
    return d

def check_pts_frmt(points):
    """
    Function to check that the input format of the points is correct

    Parameters
    ----------
    points (list) : Set of points in the form:
        [[px_1, py_1], [px_2, py_2], ..., [px_n, py_n]]
    """

    if type(points) is not list:
        raise ValueError('Is not a list')
    if len(points) == 0:
        raise ValueError('The list is empty')
    else:
        for chk in range(len(points)):
            if len(points[chk]) != 2:
                raise ValueError('An element is not in the format [px, py]')

class get_trj_vals:
    """
    Generate the values of the angles of each link for each of the points on the
    desired trajectory

    ...

    Attributes
    ----------
    init_pos : list
        Starting position for the trajectory, in form: [Px, Py]
    points : list
        Set of points in the form: [[px_1, py_1], [px_2, py_2], ..., [px_n, py_n]]

    Methods
    -------
    def trajectory:
        Generate the values of the angles of each link for the desired points

    """

    def __init__(self, init_pos, points):
        """
        Constructor

        Parameters
        ----------
        init_pos : Trajectory initial position ( [Px, Py] in R² )
        points : Desired points to generate the trajectory
                 ( [[px_1, py_1], [px_2, py_2], ..., [px_n, py_n]] )

        """

        self.pts = points
        self.ip = init_pos
        self.qs = []

    def trajectory(self):
        """
        Function to generate the values of the angles of each point

        Returns
        -------
        qs : Angle values for each link
        ip : Initial position

        """

        # Fisrt check points format
        check_pts_frmt(self.pts)

        if len(self.ip) != 2 or type(self.ip) is not list:
            raise ValueError('Initial position is not in the format [Px, Py]')

        # Gains
        kp = [[30.0, 0.0],
              [0.0, 30.0]]
        kv = [[7.0, 0.0],
              [0.0, 4.0]]
        # Initial conditios
        qi = inverse_k(self.ip[0], self.ip[1])
        _p = self.ip
        vi = [0.0, 0.0]
        ti = 0.0

        for pnt in range(len(self.pts)):
            # Check the distance with next point for change final time to point
            if euclidian_d(_p, self.pts[pnt]) > 0.05:
                # If points distance is longer, we need more time
                tf = 1.0
                # print('Flag 1')
            else:
                # If points distance is lower, we need less time
                tf = 0.6
                # print('Flag 2')

            anim_vals = pelican_robot(self.pts[pnt], kp, kv)
            q_t, v_t = anim_vals.RK4(ti, qi, vi, tf)

            if pnt == 0 or len(self.pts) == 1:
                # For generate the animation of initial point to first desired
                # point for the trajectory, to fluid way

                # To get only 100 values of initial point to first desired point
                n = len(anim_vals.values()[0]) // 100
                stp_init2dp1 = np.arange(0, len(anim_vals.values()[0]), n)
                for i2p in stp_init2dp1:
                    self.qs.append([anim_vals.values()[0][i2p][0], anim_vals.values()[0][i2p][1]])

            else:
                self.qs.append([q_t[0], q_t[1]])
                qi = [q_t[0], q_t[1]]
                vi = [v_t[0], v_t[1]]
                _p = self.pts[pnt]

        return self.qs, self.ip

class trjanim(get_trj_vals):
    """
    Generate the animation of the trajectory, the class that generates the
    values is inherited.

    Attributes
    ----------
    init_pos : list
        Starting position for the trajectory, in form: [Px, Py]
    points : list
        Set of points in the form: [[px_1, py_1], [px_2, py_2], ..., [px_n, py_n]]

    Methods
    -------
    def get_q_vals:
        Get the values to generate the animation

    def _animation_:
        Generate animation
    """

    def __init__(self, init_pos, points):
        """
        Constructor
        """

        get_trj_vals.__init__(self, init_pos, points)

    def get_q_vals(self):
        """
        Get the values to generate the animation
        """

        return get_trj_vals.trajectory(self)

    def _animation_(self):
        """
        Generate animation
        """

        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Pelican Robot: Trajectory Animation')
        # Change icon window
        plc_anim_w = plt.get_current_fig_manager()
        img = PhotoImage(file='images/pelican-robot-icon.png')
        plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

        # Pelican Robot Workspace in meteters
        ax.set_xlim((-0.6, 0.6))
        ax.set_ylim((-0.6, 0.6))
        ax.set_title('Animation Pelican: Trajectory Simulation')
        ax.set_aspect('equal')
        ax.grid('on')

        link_1, = ax.plot([], [], 'r', lw=2)
        link_2, = ax.plot([], [], 'b', lw=2)

        print('Wait while the animation is generated')

        qs, ip = self.get_q_vals()

        def init():
            """
            Get initial frame
            """

            qi = inverse_k(ip[0], ip[1])
            link_1.set_data([0.0, direct_k(qi[0], qi[1])[0][0]],
                            [0.0, direct_k(qi[0], qi[1])[0][1]])

            link_2.set_data([direct_k(qi[0], qi[1])[0][0], direct_k(qi[0], qi[1])[1][0]],
                            [direct_k(qi[0], qi[1])[0][1], direct_k(qi[0], qi[1])[1][1]])

            return link_1, link_2

        def frms(i):
            """
            Get frames of all animation
            """

            link_1.set_data([0.0, direct_k(qs[i][0], qs[i][1])[0][0]],
                            [0.0, direct_k(qs[i][0], qs[i][1])[0][1]])

            link_2.set_data([direct_k(qs[i][0], qs[i][1])[0][0],
                             direct_k(qs[i][0], qs[i][1])[2][0]],
                            [direct_k(qs[i][0], qs[i][1])[0][1],
                             direct_k(qs[i][0], qs[i][1])[2][1]])

            return link_1, link_2

        anim = animation.FuncAnimation(fig, frms, init_func=init, frames=len(qs),
                                    interval=50, blit=True)

def anim_circle_trj(r, cx, cy, stp, ip):
    """
    Animate the trajectory of the circle at the desired origin point and radius

    Parameters
    ----------
    r : Radius
    cx : x-cordinate of center
    cy : y-cordinate of center
    stp : Number of steps for circle
    ip : Initial position
    """

    c = np.linspace(0, 2 * np.pi, stp)
    points = [[r * np.cos(c[i]) + cx, r * np.sin(c[i]) + cy] for i in range(len(c))]

    trjanim(ip, points)._animation_()

if __name__ == '__main__':
    r = 0.15
    cx = 0.2
    cy = - 0.2
    stp = 50
    ip = [0.2, - 0.4]

    anim_circle_trj(r, cx, cy, stp, ip)
    plt.show()
