from typing import Union

import numpy as np
import matplotlib.pyplot as plt


def draw_displacements(domain_shape, displacements: np.ndarray, density: Union[np.ndarray, None] = None, filename: str = 'rec_disp.png'):
    nx, ny = domain_shape[0] + 1, domain_shape[1] + 1
    X, Y = np.meshgrid(range(nx), range(ny))

    if density is None:
        dx = np.array([
            [displacements[x * ny + y, 0] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)
        ])
        dy = np.array([
            [displacements[x * ny + y, 1] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)
        ])
    else:
        dx = np.array([
            [displacements[x * ny + y, 0] if density[x * ny + y] > 0.5 else 0 for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)
        ])
        dy = np.array([
            [displacements[x * ny + y, 1] if density[x * ny + y] > 0.5 else 0 for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)
        ])
    # plt.scatter(X, Y, 1, c='b')
    # plt.scatter(X + dx, Y + dy, 1, c='r')

    plt.quiver(X, Y, dx, dy)

    plt.savefig(filename)
    plt.close()