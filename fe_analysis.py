import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from plotting_utils import draw_displacements


class FiniteElementAnalysis:
    """
    This FEM implementation is based on the open-source code written by Niels Aage and Villads Egede Johansen,
    which can be found at https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python
    """
    
    def __init__(self, shape, E: float = 1, nu: float = 0.3, penal: float = 3.0):
        self.E = E
        self.nu = nu
        self.penal = penal

        self.nelx, self.nely = shape
        self.edofMat = self.create_edof_mat()
        self.iK = np.kron(self.edofMat, np.ones((8, 1))).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1, 8))).flatten()

        self.dofs = np.arange(2 * (self.nelx + 1) * (self.nely + 1))
        self.ndof = 2 * (self.nelx + 1) * (self.nely + 1)

        # setup for crane
        self.fixed = np.array(self.dofs[0: 2 * (self.nely + 1)])
        self.free = np.setdiff1d(self.dofs, self.fixed)

        self.f = np.zeros((self.ndof, 1))
        # crane
        self.f[2 * (self.nelx + 1) * (self.nely + 1) - 1, 0] = -1
        # cantilever
        # self.f[2 * (self.nelx + 1) * (self.nely + 1) - (self.nely + 1), 0] = -1
        
        self.KE = self.create_stiffmat()

    def compute_compliance(self, displacement: np.ndarray):
        return (np.dot(displacement[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE) *
                displacement[self.edofMat].reshape(self.nelx * self.nely, 8)).sum(1)

    def compute_displacement(self, density: np.ndarray):
        u = np.zeros((self.ndof, 1))
        sK = ((self.KE.flatten()[np.newaxis]).T * density).flatten(order='F')

        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[self.free, :][:, self.free]
        # Solve system

        u[self.free, 0] = spsolve(K, self.f[self.free, 0])
        return u

    # element stiffness matrix
    def create_stiffmat(self):
        k = np.array([
            1 / 2 - self.nu / 6, 1 / 8 + self.nu / 8, 
            -1 / 4 - self.nu / 12, -1 / 8 + 3 * self.nu / 8, 
            -1 / 4 + self.nu / 12, -1 / 8 - self.nu / 8,
            self.nu / 6, 1 / 8 - 3 * self.nu / 8
        ])
        KE = self.E / (1 - self.nu ** 2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])
        return KE

    def create_edof_mat(self):
        edof_mat = np.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                edof_mat[el, :] = np.array(
                    [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])
        return edof_mat


if __name__ == '__main__':
    # fe_analysis

    dom_shape = (20, 10)
    density = np.ones(dom_shape[0] * dom_shape[1], dtype=float)
    fem = FiniteElementAnalysis(shape=dom_shape)
    displ = fem.compute_displacement(density)

    displacement = displ.reshape((2, -1), order='F').T
    draw_displacements(dom_shape, displacement, density=None, filename="displacements.png")

    comp = fem.compute_compliance(displ)
    plt.matshow(comp.reshape((dom_shape[1], dom_shape[0]), order='F'))
    plt.show()
