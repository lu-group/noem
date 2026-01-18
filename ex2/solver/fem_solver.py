import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from ex2.solver.element import line_element as element

def solver(node, mesh, U_BC, U_nodeid, f_BC, f_nodeid, k_list, c_list):
    ele_node_num = 2
    # Initialize the sparse matrix directly
    row_indices = []
    col_indices = []
    data_values = []

    # Assembly of global stiffness matrix
    for k, c, ele in zip(k_list, c_list, mesh):
        ele_coords = np.array([node[i] for i in ele])
        ele_k = element.get_elek(ele_coords, k, c)
        for i in range(ele_node_num):
            for j in range(ele_node_num):
                row_indices.append(ele[i])
                col_indices.append(ele[j])
                data_values.append(ele_k[i, j])

    # Create CSR matrix from the data
    K = csr_matrix((data_values, (row_indices, col_indices)), shape=(len(node), len(node)))
    F = np.zeros(len(node))
    if len(f_BC) != 0:
        F[f_nodeid] += f_BC
    average_k = np.mean(np.abs(K.diagonal()))
    beta = 1e7 * average_k
    big_indices = np.array(U_nodeid, dtype=int)
    K[big_indices, big_indices] = beta
    F[big_indices] = beta * np.array(U_BC)
    # Solve the linear system
    T = spsolve(K, F)
    return T

if __name__ == '__main__':
    # node_num = 50
    # node = np.linspace(0, 1, node_num)
    # U_BC = [0,0]
    # U_nodeid = [0,node_num - 1]
    # f = 1
    # f_BC = []
    # f_nodeid = []
    # k_list = []
    # for i in range(len(node)):
    #     x = node[i]
    #     f_nodeid.append(i)
    #     if x < 0.5:
    #         f_BC.append((-4 * x**2 + 2 * x + 8) / (node_num - 1))
    #     else:
    #         f_BC.append((2 * x ** 2 - 3 * x - 7) / (node_num - 1))
    #
    # mesh = [[i, i + 1] for i in range(node_num - 1)]
    # for i in mesh:
    #     loc = 0.5 * (node[i[0]] + node[i[1]])
    #     if loc < 0.5:
    #         k_list.append(1)
    #     else:
    #         k_list.append(2)
    # c_list = np.ones(len(mesh)).tolist()
    # T = solver(node, mesh, U_BC, U_nodeid, f_BC, f_nodeid, k_list, c_list)
    # import matplotlib.pyplot as plt
    # plt.plot(node, T)
    # plt.show()

    # node_num = 501
    # node = np.linspace(0, 10, node_num)
    # U_BC = [0.1635]
    # U_nodeid = [0]
    # f = 1
    # f_BC = []
    # f_nodeid = []
    # k_list = []
    # for i in range(len(node)):
    #     x = node[i]
    #     f_nodeid.append(i)
    #     x = x % 1
    #     print(x)
    #     f_BC.append(x * (1-x) * 10 / (node_num - 1))
    #
    # mesh = [[i, i + 1] for i in range(node_num - 1)]
    # c_list = np.ones(len(mesh)).tolist()
    # k_list = np.ones(len(mesh)).tolist()
    # T = solver(node, mesh, U_BC, U_nodeid, f_BC, f_nodeid, k_list, c_list)
    # import matplotlib.pyplot as plt
    # plt.grid(True)
    # plt.plot(node, T)
    # plt.show()

    node_num = 1001
    L = 10
    node = np.linspace(0, L, node_num)
    U_BC = [0.,0.]
    U_nodeid = [0, node_num - 1]
    f = 0.5
    f_BC = []
    f_nodeid = []
    k_list = []
    for i in range(len(node)):
        x = node[i]
        f_nodeid.append(i)
        x = x % 1
        print(x)
        f_BC.append(f * L / (node_num - 1))

    mesh = [[i, i + 1] for i in range(node_num - 1)]
    c_list = np.zeros(len(mesh)).tolist()
    ele_loc = [0.5 * (node[i] + node[i + 1]) for i in range(len(mesh))]
    ele_loc = np.array(ele_loc)
    wave_num = 1
    k_list = 0.5 * np.sin(2 * np.pi * ele_loc * wave_num) + 0.8
    T = solver(node, mesh, U_BC, U_nodeid, f_BC, f_nodeid, k_list, c_list)
    print(max(T))

    import matplotlib.pyplot as plt

    plt.grid(True)
    plt.plot(node, T)
    plt.show()