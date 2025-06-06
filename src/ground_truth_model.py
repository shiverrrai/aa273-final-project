import numpy as np
import math
import scipy.integrate
import constants as consts


def dynamics(t, x):
    '''
    :param x: state vector
    :return: time derivative of state vector
    '''
    rx, ry, rz, vx, vy, vz = x
    v = np.array([vx, vy, vz])
    speed = np.linalg.norm(v)
    F_gravity = np.array([0, 0, consts.m * consts.g])
    F_drag = -0.5 * consts.rho * consts.Cd * (
            math.pi * math.pow(consts.r, 2)) * speed * v
    a = (F_gravity + F_drag) / consts.m
    dxdt = np.array([vx, vy, vz, a[0], a[1], a[2]])
    return dxdt


def bounce(t, x):
    '''
    An event callable to be used in solve_ivp function.
    Terminates integration when z-component of ball position reaches zero,
    triggered when z-component goes from positive to negative.
    '''
    # return z-component of ball position
    return x[2]


bounce.terminal = True
bounce.direction = -1


class SystemModel:

    def __init__(self, x0, total_time, dt):
        '''
        :param x0: initial state vector
        :param total_time: total time to run simulation
        :param dt: time step
        '''
        assert x0.shape == (6,), "Initial state vector must have 6 dimensions"
        self.x0 = x0
        self.total_time = total_time
        self.dt = dt
        self.x_impact = None

    def reset(self):
        self.x_impact = None

    def run_sim(self, bounces=2):
        '''
        :param bounces: number of bounces to run simulation for
        :return: a time and state vector of the tennis ball
        '''
        t_hist = []
        x_hist = []
        t0 = 0
        x0 = self.x0
        for _ in range(bounces):
            # solves ball trajectory until a bounce event occurs
            sol = scipy.integrate.solve_ivp(
                fun=dynamics,
                t_span=(t0, self.total_time),
                y0=x0,
                max_step=self.dt,
                events=bounce,
            )
            t_hist.extend(sol.t)
            x_hist.extend(sol.y.T)

            # handle bounce event
            if sol.status == 1 and self.x_impact is None:
                # copy state vector when bounce occurred
                x, y, z, vx, vy, vz = sol.y[:, -1]
                # store ball's first impact location
                self.x_impact = sol.y[:, -1]

                # exit sim if ball has almost stopped
                if abs(vz) < 0.1:
                    break

                # model restitution due to impact and change sign of vz
                vz = -consts.e * vz

                # update initial conditions for next iteration\
                # small z offset to avoid event loop
                x0 = [x, y, 0.001, vx, vy, vz]
                t0 = sol.t[-1]

            # the solver completed successfully or integration failed. In either
            # case, end the simulation.
            else:
                break

        return np.asarray(t_hist), np.asarray(x_hist)
