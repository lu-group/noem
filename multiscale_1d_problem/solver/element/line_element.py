import numpy as np
def get_elek(ele_coords, k, c):
    L = abs(ele_coords[1] - ele_coords[0])
    K_diff = k * np.array([[1, -1],
                               [-1, 1]]) / L

    # Stiffness matrix due to potential terms (second part of the equation)
    K_pot = c * np.array([[2, 1],
                          [1, 2]]) * L / 6

    # Total stiffness matrix
    K_total = K_diff + K_pot
    return K_total
