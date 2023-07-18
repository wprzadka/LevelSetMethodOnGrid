import matplotlib.pyplot as plt
import numpy as np

from fe_analysis import FiniteElementAnalysis
from holes_initialization import initialize_phi_func, domain_from_phi, compute_sign_dist, \
    initialize_phi_func_with_padding
from plotting_utils import draw_displacements
from upwind_scheme import UpwindScheme
from scipy.signal import convolve2d


class LeveLSetOptimization:

    def __init__(self, shape: tuple, lag_mult: float, dt: float, void: float):
        self.shape = shape
        self.lag = lag_mult
        self.dt = dt
        self.void = void

    def optimize(self, niters: int):

        scheme = UpwindScheme(self.dt)
        fem = FiniteElementAnalysis(self.shape)

        phi = initialize_phi_func(self.shape, (16, 6), 0.5)
        density = domain_from_phi(phi, low=self.void)

        plt.matshow(phi)
        plt.title("phi initial")
        plt.colorbar()
        plt.show()

        for iter in range(1, niters + 1):

            plt.matshow(density, cmap='gray_r')
            plt.title("Density")
            plt.savefig(f'dens/dens{iter}.png')
            plt.show()

            displ = fem.compute_displacement(density.flatten(order='F'))

            displacement = displ.reshape((2, -1), order='F').T
            draw_displacements(dom_shape, displacement, density=None, filename=f"displ/displ{iter}.png")

            compliance = fem.compute_compliance(displ)
            compliance = compliance.reshape((dom_shape[1], dom_shape[0]), order='F')
            # remove the parts with no density
            compliance *= density

            vel = compliance - self.lag

            print(f'iteration: {iter} cost: {np.sum(vel)}')

            plt.matshow(vel)
            plt.title("velocity")
            plt.savefig(f'velocity/vel{iter}.png')
            plt.show()

            # conv_filter = 1 / 6 * np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]])
            conv_filter = 1 / 256 * np.array([
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1]
            ])
            vel = convolve2d(vel, conv_filter, boundary='pad', fillvalue=0, mode='same')

            plt.matshow(vel)
            plt.title("velocity after filtering")
            plt.savefig(f'velocity/vel{iter}_f.png')
            plt.show()

            # TODO check
            # phi = scheme.update(phi, vel)
            for i in range(20):
                phi = scheme.update_test(phi, vel)

            plt.matshow(phi)
            plt.title("phi")
            plt.colorbar()
            plt.show()

            density = domain_from_phi(phi, low=self.void)

            if iter % 5 == 0 and iter > 0:
                phi = compute_sign_dist(density)

                plt.matshow(phi)
                plt.title("phi after reinitialization")
                plt.colorbar()
                plt.show()
        return density, phi


if __name__ == '__main__':

    dom_shape = (160, 80)

    opt = LeveLSetOptimization(shape=dom_shape, lag_mult=0.09, dt=0.1, void=1e-4)
    density, phi = opt.optimize(100)

    plt.matshow(density)
    plt.show()
