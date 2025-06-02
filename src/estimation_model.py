import numpy as np
import constants as consts

class FlightModel:
    def f(self, state, dt):
        """
        Compute state dynamics for ball in flight.

        :param state: 6-dimensional state vector (3 position
        states, 3 velocity states): [x, y, z, xdot, ydot, zdot]˜
        """
        x, y, z, xdot, ydot, zdot = state
        x += dt * xdot
        y += dt * ydot
        z += dt * zdot + 0.5 * consts.g * (dt ** 2)
        zdot += consts.g * dt
        return np.array([x, y, z, xdot, ydot, zdot])

    def A(self, dt):
        """
        Compute the linearized state dynamics of ball in flight.
        :return: 6x6 matrix
        """
        A = np.eye(6)
        A[0:3, 3:] = dt * np.eye(3)
        return A


class BounceModel:

    def f(self, state, dt):
        """
        Compute state dynamics for ball during bounce.

        :param state: 6-dimensional state vector (3 position
        states, 3 velocity states): [x, y, z, xdot, ydot, zdot]˜
        """
        x, y, z, xdot, ydot, zdot = state
        x += dt * xdot
        y += dt * ydot
        z += dt * zdot + 0.5 * consts.g * (dt ** 2)
        zdot = -consts.e * zdot + consts.g * dt
        return np.array([x, y, z, xdot, ydot, zdot])

    def A(self, dt):
        """
        Compute the linearized state dynamics of ball during bounce.
        :return: 6x6 matrix
        """
        A = np.eye(6)
        A[0:3, 3:] = dt * np.eye(3)
        A[5, 5] *= -consts.e
        return A