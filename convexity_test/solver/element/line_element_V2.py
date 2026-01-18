import numpy as np
def get_elek(ele_coords, k, c, u1, u2):
    L = abs(ele_coords[1] - ele_coords[0])
    K_diff = k * np.array([[1, -1],
                               [-1, 1]]) / L

    # Stiffness matrix due to potential terms (second part of the equation)
    K_pot = c * np.array([[2, 1],
                          [1, 2]]) * L / 6

    K_grad = - np.array([[u1 * 2, -2],[-2, u2 * 2]]) / L
    # Total stiffness matrix
    K_total = K_diff + K_pot - K_grad
    return K_total

def get_elef(ele_coords, k, c, u1, u2):
    L = abs(ele_coords[1] - ele_coords[0])
    F = -np.array([2 * u1 - 2 * u2, 2 * u2 - 2 * u1]) / L
    return F
