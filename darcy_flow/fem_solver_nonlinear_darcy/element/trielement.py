import numpy as np

def get_elek(coords, k):
    """
    Calculate the element stiffness matrix for a triangular element.
    coords : numpy.ndarray
        An array of shape (3, 2) containing the coordinates of the element vertices
    k : float
        Thermal conductivity
    Returns:
    numpy.ndarray
        The element stiffness matrix (3x3)
    """
    # Define shape functions derivatives for a triangular element
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    A = 0.5 * np.linalg.det(np.array([[1, x1, y1],
                                      [1, x2, y2],
                                      [1, x3, y3]]))
    if A < 0:
        raise ValueError("The element vertices are not in the correct order")

    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    # Stiffness matrix computation
    K = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K[i, j] = k * (b[i] * b[j] + c[i] * c[j]) / (4 * A)

    return K

def get_elef_tri(coords, q):
    """
    Calculate the element load vector for a triangular element.
    coords : numpy.ndarray
        An array of shape (3, 2) containing the coordinates of the element vertices
    q : float
        Heat source per unit volume
    Returns:
    numpy.ndarray
        The element load vector (3,)
    """
    # Compute the area of the triangle
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    A = 0.5 * np.linalg.det(np.array([[1, x1, y1],
                                      [1, x2, y2],
                                      [1, x3, y3]]))

    # Load vector computation
    f = np.full(3, q * A / 3)  # Equal contribution to each node

    return f

if __name__ == '__main__':
    a = 1
    coords_tri = np.array([[0, 0], [a, 0], [a, a], [0, a]])
    k = 1
    K_tri1 = get_elek(coords_tri[:3,:], k)
    K_tri2 = get_elek(coords_tri[1:,:], k)
    K = np.zeros((4, 4))
    K[:3, :3] += K_tri1
    K[1:, 1:] -= K_tri2
    print("Stiffness matrix for triangular element:")
    print(K)

    # from quaelement import get_elek as get_elek_quad
    # coords_quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    # K_quad = get_elek_quad(coords_quad, k)
    # print("Stiffness matrix for quadrilateral element:")
    # print(K_quad)

