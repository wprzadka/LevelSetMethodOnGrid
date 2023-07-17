import matplotlib.pyplot as plt
import numpy as np
import skfmm
import json

# def compute_initial_phi(grid: np.ndarray, holes: np.ndarray):
#
#
# def get_unif_centers(shape: tuple, holes_per_dim: tuple):
#     nelx, nely = shape
#     dx = nelx / holes_per_dim[0]
#     dy = nely / holes_per_dim[1]
#
#     holes_x = np.linspace(dx / 2, nelx - dx / 2, holes_per_dim[0])
#     holes_y = np.linspace(dy / 2, nely - dy / 2, holes_per_dim[1])
#
#     return np.array([[x, y] for x in holes_x for y in holes_y])


# Osher S., Fedkiw R., Level set methods and dynamic implicit surfaces, Applied
# Mathematical Sciences, 153, Springer-Verlag, New York (2003)
def initialize_phi_func(shape: tuple, holes_per_axis: tuple, radius: float):
    phi = generate_cosine_func(shape, holes_per_axis, radius)
    phi = np.where(phi > 0, 0.1, -0.1)
    phi = skfmm.distance(phi, dx=1)
    return phi


def initialize_phi_func_with_padding(shape: tuple, holes_per_axis: tuple, radius: float, padding: int):
    reduced_shape = (shape[0] - 2 * padding, shape[1] - 2 * padding)
    phi = generate_cosine_func(reduced_shape, holes_per_axis, radius)
    phi = np.where(phi > 0, 0.1, -0.1)
    phi = np.pad(phi, padding, constant_values=-0.1)
    phi = skfmm.distance(phi, dx=1)
    return phi


def generate_cosine_func(shape: tuple, holes_per_axis: tuple, radius: float):
    dom_x = np.linspace(0, 1, shape[0])
    dom_y = np.linspace(0, 1, shape[1])
    X, Y = np.meshgrid(dom_x, dom_y)
    # print(X.shape, Y.shape)
    # phi = np.array([[-np.cos] for x in dom_x] for y in dom_y)
    phi = -np.cos(X * holes_per_axis[0] * np.pi) * np.cos(Y * holes_per_axis[1] * np.pi) + radius - 1
    return phi

# d(phi)/dt + sign(phi0)(|grad(phi)|-1) = 0 with
# phi(t=0,x) = phi_0(x)
def compute_sign_dist(density: np.ndarray):
    domain = np.where(density > 0.5, -0.1, 0.1)
    sd = skfmm.distance(domain, dx=1)
    return sd


def domain_from_phi(phi: np.ndarray, low: float = 1e-9):
    return np.where(phi < 0, 1., low)


if __name__ == '__main__':
    shape = (80, 40)
    phi = initialize_phi_func(shape, (12, 4), 0.7)

    plt.matshow(phi)
    plt.show()

    plt.matshow(domain_from_phi(phi))
    plt.show()

    with open('dom.txt', 'w') as f:
        a=domain_from_phi(phi)
        txt = json.dumps(list(list(r) for r in a))
        f.write(txt)