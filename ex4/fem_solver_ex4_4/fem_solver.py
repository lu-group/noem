import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import ex4.fem_solver_ex4_4.element.quaelement as element


def get_node(x_len, y_len, num_x, num_y):
    # Generate node coordinates correctly
    node = [[j * x_len / num_x, i * y_len / num_y] for i in range(num_y + 1) for j in range(num_x + 1)]
    return node

def get_mesh(num_x, num_y):
    # Generate mesh connectivity correctly
    mesh = []
    for i in range(num_y):
        for j in range(num_x):
            mesh.append([i * (num_x + 1) + j, i * (num_x + 1) + j + 1, (i + 1) * (num_x + 1) + j + 1, (i + 1) * (num_x + 1) + j])
    return mesh

def solver(node, mesh, T_BC, T_nodeid, q_BC, q_nodeid, k_list, element_type='quad4', is_logging=True):
    ele_node_num = 4
    num_nodes = len(node)
    appq = np.zeros(num_nodes)
    unbq = np.zeros(num_nodes)
    T = np.zeros(num_nodes)
    for i in range(len(q_nodeid)):
        nodeid = q_nodeid[i]
        appq[int(nodeid)] = q_BC[i]
    unbq += appq
    iter = 0
    if len(T_BC) != 0:
        tol_criteria_type = 1
    while True:

        row_indices = []
        col_indices = []
        data_values = []

        # Assembly of global stiffness matrix
        for z in range(len(mesh)):
            ele = mesh[z]
            tk = k_list[z]
            ele_coords = np.array([node[i] for i in ele])
            tT = np.array([T[i] for i in ele])
            ele_k = element.get_elek(ele_coords, tk, tT)
            for i in range(ele_node_num):
                for j in range(ele_node_num):
                    row_indices.append(ele[i])
                    col_indices.append(ele[j])
                    data_values.append(ele_k[i, j])

        # # Create CSR matrix from the data
        K = csr_matrix((data_values, (row_indices, col_indices)), shape=(len(node), len(node)))

        equal_unbq = unbq
        # Apply temperature boundary conditions
        average_k = np.mean(np.max(K.diagonal()))
        beta = 1e7 * average_k
        big_indices = np.array(T_nodeid, dtype=int)
        K[big_indices, big_indices] = beta
        if iter == 0:
            equal_unbq[big_indices] = beta * np.array(T_BC)
        else:
            equal_unbq[big_indices] = 0

        delta_T = spsolve(K, equal_unbq)
        # delta_T = np.linalg.solve(K, equal_unbq)
        T += delta_T

        updated_q = np.zeros(num_nodes)
        # Solve the system of equations
        for z in range(len(mesh)):
            ele = mesh[z]
            tk = k_list[z]
            ele_coords = np.array([node[i] for i in ele])
            tT = np.array([T[i] for i in ele])
            ele_k = element.get_elek(ele_coords, tk, tT)
            ele_T = np.array([T[i] for i in ele])
            ele_q = ele_k @ ele_T
            updated_q[ele] += ele_q

        # apply BC
        for i in range(len(T_nodeid)):
            updated_q[T_nodeid[i]] = 0

        delta_q = appq - updated_q

        if tol_criteria_type == 1:
            tol_criteria = np.linalg.norm(delta_T) / np.linalg.norm(T)
            if is_logging:
                print("magnitude of delta_T:", tol_criteria)
            if tol_criteria < 1e-3:
                break
        else:
            tol_criteria = np.linalg.norm(delta_q) / np.linalg.norm(appq)
            if is_logging:
                print("magnitude of delta_q:", tol_criteria)
            if tol_criteria < 1e-3:
                break

        iter += 1
        if iter > 9:
            break
        unbq = delta_q
    return T

def get_fourside_nodeid(num_x, num_y):
    left_nodeid = [i * (num_x + 1) for i in range(num_y + 1)]
    right_nodeid = [(i + 1) * (num_x + 1) - 1 for i in range(num_y + 1)]
    top_nodeid = [i + (num_x + 1) * num_y for i in range(num_x + 1)]
    bottom_nodeid = [i for i in range(num_x + 1)]
    return left_nodeid, right_nodeid, top_nodeid, bottom_nodeid

if __name__ == '__main__':
    # x_len = y_len = 0.8
    # num_x = num_y = 10
    # node = get_node(x_len, y_len, num_x, num_y)
    # mesh = get_mesh(num_x, num_y)
    #
    # # Define boundary conditions
    # T_nodeid = [i * (num_x + 1) for i in range(num_y + 1)]
    # T_mag = 100
    # T_BC = np.zeros(len(T_nodeid)) + T_mag  # Constant temperature boundary condition
    # tT_nodeid = [(i + 1) * (num_x + 1) - 1 for i in range(num_y + 1)]
    # T_nodeid += tT_nodeid
    # tT_BC = np.zeros(len(tT_nodeid))  # Zero temperature boundary condition
    # T_BC = np.concatenate((T_BC, tT_BC))
    # q_nodeid = [(i + 1) * (num_x + 1) - 1 for i in range(num_y + 1)]
    # q_mag = 0
    # q_BC = np.full(len(q_nodeid), q_mag / y_len * (y_len / num_y))  # Evenly distribute heat flux
    # q_BC[0] /= 2
    # q_BC[-1] /= 2

    x_len = 1.0
    y_len = 1.0
    num_x = num_y = 4
    node = get_node(x_len, y_len, num_x, num_y)
    mesh = get_mesh(num_x, num_y)

    # Define boundary conditions
    left_nodeid = [i * (num_x + 1) for i in range(num_y + 1)]
    right_nodeid = [(i + 1) * (num_x + 1) - 1 for i in range(num_y + 1)]
    top_nodeid = [i + (num_x + 1) * num_y for i in range(num_x + 1)]
    bottom_nodeid = [i for i in range(num_x + 1)]

    T_nodeid = left_nodeid + right_nodeid
    T_BC1 = np.ones(len(left_nodeid))
    T_BC2 = np.zeros(len(left_nodeid)) - 1
    T_BC = np.concatenate((T_BC1, T_BC2))

    T = solver(node, mesh, T_BC, T_nodeid, q_BC=[], q_nodeid=[], k_list=np.zeros(len(mesh)), element_type='quad4')

    # Print temperature for each node
    for y in range(num_y + 1):
        for x in range(num_x + 1):
            idx = y * (num_x + 1) + x
            print(f"({node[idx][0]:.2f}, {node[idx][1]:.2f}) T={T[idx]:.2f}", end=' ')
        print()

    # Plotting
    plt.figure()
    plt.tricontourf([n[0] for n in node], [n[1] for n in node], T, levels=100)
    plt.colorbar()
    plt.title("Temperature Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    # adjust the aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    # plot the contour lines
    # plt.tricontour([n[0] for n in node], [n[1] for n in node], T, colors='k', levels=10)
    plt.show()
