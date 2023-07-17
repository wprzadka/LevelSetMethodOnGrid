import matplotlib.pyplot as plt
import numpy as np


class UpwindScheme:

    def __init__(self, dt: float):
        self.dt = dt

    def update(self, phi: np.ndarray, velocity: np.ndarray):
        negVel = np.minimum(velocity, 0)
        posVel = np.maximum(velocity, 0)

        # TODO check
        # self.dt = 1 / np.max(np.abs(velocity))
        # print(self.dt)

        forward_x = np.empty_like(phi)
        forward_x[1:, :] = phi[1:, :] - phi[:-1, :]
        forward_x[0, :] = forward_x[1, :]

        backward_x = np.empty_like(phi)
        backward_x[:-1, :] = phi[1:, :] - phi[:-1, :]
        backward_x[-1, :] = backward_x[-2, :]

        forward_y = np.empty_like(phi)
        forward_y[:, 1:] = phi[:, 1:] - phi[:, :-1]
        forward_y[:, 0] = forward_y[:, 1]

        backward_y = np.empty_like(phi)
        backward_y[:, :-1] = phi[:, 1:] - phi[:, :-1]
        backward_y[:, -1] = backward_y[:, -2]

        return phi - self.dt * (
                negVel * (self.g_minus(forward_x, backward_x) + self.g_minus(forward_y, backward_y)) +
                posVel * self.g_plus(forward_x, backward_x) + self.g_plus(forward_y, backward_y)
        )

    def g_plus(self, d_plus: np.ndarray, d_minus: np.ndarray) -> np.ndarray:
        return np.sqrt(np.minimum(d_plus, 0) ** 2 + np.maximum(d_minus, 0) ** 2)

    def g_minus(self, d_plus: np.ndarray, d_minus: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(d_plus, 0) ** 2 + np.minimum(d_minus, 0) ** 2)

    def update_test(self, phi: np.ndarray, vel: np.ndarray):

        # TODO check
        self.dt = 0.1 / np.max(np.abs(vel))

        dpx = np.roll(phi, -1, axis=1) - phi
        dpx[:, -1] = dpx[:, -2]
        dmx = phi - np.roll(phi, 1, axis=1)
        dmx[:, 0] = dmx[:, 1]
        dpy = np.roll(phi, -1, axis=0) - phi
        dpy[-1, :] = dpy[-2, :]
        dmy = phi - np.roll(phi, 1, axis=0)
        dmy[0, :] = dmy[1, :]

        return phi \
            - self.dt * np.minimum(vel, 0) * np.sqrt(
                np.minimum(dmx, 0) ** 2 + np.maximum(dpx, 0) ** 2
                + np.minimum(dmy, 0) ** 2 + np.maximum(dpy, 0) ** 2
            ) \
            - self.dt * np.maximum(vel, 0) * np.sqrt(
                np.maximum(dmx, 0) ** 2 + np.minimum(dpx, 0) ** 2
                + np.maximum(dmy, 0) ** 2 + np.minimum(dpy, 0) ** 2
            )


if __name__ == '__main__':
    scheme = UpwindScheme(dt=0.5)

    dom_x = np.linspace(0, 1)
    dom_y = np.linspace(0, 1)
    X, Y = np.meshgrid(dom_x, dom_y)
    # phi = np.cos(X * np.pi * 3) * Y
    init_phi = -np.cos(X * 3 * np.pi) * np.cos(Y * 3 * np.pi) + 0.5 - 1

    plt.matshow(init_phi)
    plt.show()

    vel = X * Y * 100 - 50
    phi = scheme.update(init_phi, vel)
    phi2 = scheme.update_test(init_phi, vel)

    plt.matshow(phi - init_phi)
    plt.colorbar()
    plt.show()

    plt.matshow(phi2 - init_phi)
    plt.colorbar()
    plt.show()
