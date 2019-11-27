import numpy as np
import matplotlib.pyplot as plt

class Robot2DOF(object):
    def __init__(self, function1, function2):
        self.f = function1
        self.g = function2
        self.ts = []
        self.us = []
        self.vs = []
        self.i = 0

    def RK(self, ti, uin, vin, done, h):
        self.ts.append(ti)

        self.us.append(uin)
        ui = self.us[self.i]

        self.vs.append(vin)
        vi = self.vs[self.i]

        while True:
            K1 = self.f(vi)
            m1 = self.g(vi, ui)

            K2 = self.f(vi + np.dot(m1, h / 2))
            m2 = self.g(vi + np.dot(m1, h / 2), ui + np.dot(K1, h / 2))

            K3 = self.f(vi + np.dot(m2, h / 2))
            m3 = self.g(vi + np.dot(m2, h / 2), ui + np.dot(K2, h / 2))

            K4 = self.f(vi + np.dot(m3, h))
            m4 = self.g(vi + np.dot(m3, h), ui + np.dot(K3, h))

            ui += np.dot((h / 6), (K1 + np.dot(2, K2, K3) + K4))
            uitemp = [ui[0], ui[1]]
            self.us.append(uitemp)

            vi += np.dot((h / 6), (m1 + np.dot(2, m2, m3) + m4))
            vitemp = [vi[0], vi[1]]
            self.vs.append(vitemp)

            ti += h
            self.ts.append(ti)

            self.i += 1
            if ti >= done:
                break

        return self.us, self.vs, self.ts, self.i

    def error(self):
        qt1 = []
        qt2 = []

        for qs in range(0, len(self.us)):
            q1 = self.us[qs][0]
            q2 = self.us[qs][1]
            qd = [[np.pi / 10, np.pi / 30]]
            t1 = qd[0][0] - q1
            t2 = qd[0][1] - q2
            qt = [t1, t2]

            qt1.append(qt[0])
            qt2.append(qt[1])

        plt.style.use('dark_background')
        plt.title("Graph of $ \\tilde{q} $")
        plt.plot(self.ts, qt1, "b--", label="$ \\tilde{q_1} $")
        plt.plot(self.ts, qt2, "r--", label="$ \\tilde{q_2} $")
        plt.legend()
        plt.grid()
        plt.xlabel("$ t $")
        plt.ylabel("$ \\tilde{q} $")