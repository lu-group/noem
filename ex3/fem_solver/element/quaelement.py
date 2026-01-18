import numpy as np

def get_elek(coords, k):
    """
    Calculate the element stiffness matrix for a rectangular element.
    coords : numpy.ndarray
        An array of shape (4, 2) containing the coordinates of the element vertices
    k : float
        Thermal conductivity
    Returns:
    numpy.ndarray
        The element stiffness matrix (4x4)
    """
    gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
    gauss_weights = [1, 1]
    K = np.zeros((4, 4))

    # Define shape functions N1 to N4
    # Define derivatives of shape functions w.r.t xi and eta
    def get_dN_dxi(xi, eta, i):
        if i == 0:
            return -0.25 * (1 - eta)
        elif i == 1:
            return 0.25 * (1 - eta)
        elif i == 2:
            return 0.25 * (1 + eta)
        elif i == 3:
            return -0.25 * (1 + eta)

    def get_dN_deta(xi, eta, i):
        if i == 0:
            return -0.25 * (1 - xi)
        elif i == 1:
            return -0.25 * (1 + xi)
        elif i == 2:
            return 0.25 * (1 + xi)
        elif i == 3:
            return 0.25 * (1 - xi)

    def get_J(loc, coords):
        J = np.zeros((2, 2))
        for i in range(4):
            J[0, 0] += get_dN_dxi(loc[0], loc[1], i) * coords[i, 0]
            J[0, 1] += get_dN_deta(loc[0], loc[1], i) * coords[i, 0]
            J[1, 0] += get_dN_dxi(loc[0], loc[1], i) * coords[i, 1]
            J[1, 1] += get_dN_deta(loc[0], loc[1], i) * coords[i, 1]
        return J

    for a in range(2):
        for b in range(2):
            xi = gauss_points[a]
            eta = gauss_points[b]
            J = get_J([xi, eta], coords)
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)  # Compute the inverse of the Jacobian matrix

            for i in range(4):
                for j in range(4):
                    # Calculate global derivatives
                    dNi_xi, dNi_eta = get_dN_dxi(xi, eta, i), get_dN_deta(xi, eta, i)
                    dNj_xi, dNj_eta = get_dN_dxi(xi, eta, j), get_dN_deta(xi, eta, j)

                    # Transform gradients to global coordinates
                    dNi_dx = invJ[0, 0] * dNi_xi + invJ[0, 1] * dNi_eta
                    dNi_dy = invJ[1, 0] * dNi_xi + invJ[1, 1] * dNi_eta
                    dNj_dx = invJ[0, 0] * dNj_xi + invJ[0, 1] * dNj_eta
                    dNj_dy = invJ[1, 0] * dNj_xi + invJ[1, 1] * dNj_eta

                    # Compute stiffness matrix contributions
                    K[i, j] += k * (dNi_dx * dNj_dx + dNi_dy * dNj_dy) * detJ * gauss_weights[a] * gauss_weights[b]

    return K

import numpy as np


def get_elef(coords, q):
    """
    Calculate the element load vector for a rectangular element.
    coords : numpy.ndarray
        An array of shape (4, 2) containing the coordinates of the element vertices
    q : float
        Heat source per unit volume
    Returns:
    numpy.ndarray
        The element load vector (4,)
    """
    gauss_points = [-1/np.sqrt(3), 1/np.sqrt(3)]
    gauss_weights = [1, 1]
    f = np.zeros(4)

    def get_J(loc, coords):
        """
        Calculate the Jacobian matrix for a given location in the local coordinates.
        """
        J = np.zeros((2, 2))
        for i in range(4):
            xi, eta = loc
            J[0, 0] += (0.25 * (-1 if i % 2 == 0 else 1) * (1 - eta if i < 2 else 1 + eta)) * coords[i, 0]
            J[0, 1] += (0.25 * (-1 if i % 2 == 0 else 1) * (1 - eta if i < 2 else 1 + eta)) * coords[i, 1]
            J[1, 0] += (0.25 * (1 - xi if i % 3 == 0 else 1 + xi)) * coords[i, 0]
            J[1, 1] += (0.25 * (1 - xi if i % 3 == 0 else 1 + xi)) * coords[i, 1]
        return J

    # Define shape functions
    def N(xi, eta, i):
        if i == 0:
            return 0.25 * (1 - xi) * (1 - eta)
        elif i == 1:
            return 0.25 * (1 + xi) * (1 - eta)
        elif i == 2:
            return 0.25 * (1 + xi) * (1 + eta)
        elif i == 3:
            return 0.25 * (1 - xi) * (1 + eta)

    for a in range(2):
        for b in range(2):
            xi = gauss_points[a]
            eta = gauss_points[b]
            J = get_J([xi, eta], coords)
            detJ = np.linalg.det(J)

            for i in range(4):
                f[i] += N(xi, eta, i) * q * detJ * gauss_weights[a] * gauss_weights[b]

    return f

if __name__ == '__main__':
    a = 2
    coords = np.array([[0, 0], [a, 0], [a, a], [0, a]])
    k = 1
    q = 1
    K = get_elek(coords, k)
    F = get_elef(coords, q)
    print(K)
    print(F)