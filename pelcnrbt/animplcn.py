# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tkinter import PhotoImage
import argparse
from probot import *

def check_pts_frmt(points):
    if type(points) is not list:
        raise ValueError('Is not a list')
    if len(points) == 0:
        raise ValueError('The list is empty')
    else:
        for chk in range(len(points)):
            if len(points[chk]) != 2:
                raise ValueError('An element is not in the format [Px, Py]')

class get_trj_vals:
    def __init__(self, init_pos, points):
        self.pts = points
        self.ip = init_pos
        self.qs = []

    def trajectory(self):
        check_pts_frmt(self.pts)
        if len(self.ip) != 2 or type(self.ip) is not list:
            raise ValueError('Initial position is not in the format [Px, Py]')

        # Gains
        kp = [[30.0, 0.0],
              [0.0, 30.0]]
        kv = [[7.0, 0.0],
              [0.0, 3.0]]
        # Initial conditios
        qi = pelican_robot.inverse('', self.ip[0], self.ip[1])
        vi = [0.0, 0.0]
        ti = 0.0
        tf = 1.0

        for pnt in range(len(self.pts)):
            anim_vals = pelican_robot(self.pts[pnt], kp, kv)
            q_t, v_t = anim_vals.RK4(ti, qi, vi, tf)
            if pnt == 0 or len(self.pts) == 1:
                stp_init2dp1 = np.arange(0, len(anim_vals.values()[0]), 10)
                for i2p in stp_init2dp1:
                    self.qs.append([anim_vals.values()[0][i2p][0], anim_vals.values()[0][i2p][1]])

            else:
                self.qs.append([q_t[0], q_t[1]])
                qi = [q_t[0], q_t[1]]
                vi = [v_t[0], v_t[1]]

        return self.qs, self.ip

class trjanim(get_trj_vals):
    def __init__(self, init_pos, points):
        get_trj_vals.__init__(self, init_pos, points)

    def get_q_vals(self):
        return get_trj_vals.trajectory(self)

    def _animation_(self):
        fig, ax = plt.subplots()
        fig.canvas.set_window_title('Pelican Robot: Trajectory Animation')
        plc_anim_w = plt.get_current_fig_manager()
        img = PhotoImage(file='images\pelican-robot-icon.png')
        plc_anim_w.window.tk.call('wm', 'iconphoto', plc_anim_w.window._w, img)

        ax.set_xlim((-0.6, 0.6))
        ax.set_ylim((-0.6, 0.6))
        ax.set_title('Animation Pelican: Trajectory Simulation')
        ax.set_aspect('equal')
        ax.grid('on')

        link_1, = ax.plot([], [], 'r', lw=2)
        link_2, = ax.plot([], [], 'b', lw=2)

        qs, ip = self.get_q_vals()

        def init():
            qi = pelican_robot.inverse('', ip[0], ip[1])
            link_1.set_data([0.0, plcn_drct_kinematic(qi[0], qi[1])[0][0]],
                            [0.0, plcn_drct_kinematic(qi[0], qi[1])[0][1]])

            link_2.set_data([plcn_drct_kinematic(qi[0], qi[1])[0][0], plcn_drct_kinematic(qi[0], qi[1])[1][0]],
                            [plcn_drct_kinematic(qi[0], qi[1])[0][1], plcn_drct_kinematic(qi[0], qi[1])[1][1]])

            return link_1, link_2

        def frms(i):
            link_1.set_data([0.0, plcn_drct_kinematic(qs[i][0], qs[i][1])[0][0]],
                            [0.0, plcn_drct_kinematic(qs[i][0], qs[i][1])[0][1]])

            link_2.set_data([plcn_drct_kinematic(qs[i][0], qs[i][1])[0][0],
                             plcn_drct_kinematic(qs[i][0], qs[i][1])[2][0]],
                            [plcn_drct_kinematic(qs[i][0], qs[i][1])[0][1],
                             plcn_drct_kinematic(qs[i][0], qs[i][1])[2][1]])

            return link_1, link_2

        print('Wait while the animation is generated with a total of {} frames...'.format(len(qs)))
        anim = animation.FuncAnimation(fig, frms, init_func=init, frames=len(qs),
                                    interval=50, blit=True)

def anim_circle_trj(r, cx, cy, stp, ip):
    c = np.linspace(0, 2 * np.pi, stp)
    points = [[r * np.cos(c[i]) + cx, r * np.sin(c[i]) + cy] for i in range(len(c))]

    trjanim(ip, points)._animation_()

if __name__ == '__main__':
    r = 0.15
    cx = 0.2
    cy = -0.2
    stp = 50
    ip = [0.2, -0.4]

    anim_circle_trj(r, cx, cy, stp, ip)
    plt.show()
