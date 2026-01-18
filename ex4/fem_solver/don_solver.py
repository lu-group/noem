# This modulus supports the analysis domain containing multiple don elements.

# from heat_transfer.fem_solver.element import trielement as trielement
import ex4.fem_solver.element.quaelement as element
import numpy as np
import torch
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
def one_solver(node, mesh, T_BC, T_nodeid, q_BC, q_nodeid, T_oneBC_nodeid_list, don_element,
               k_fem_list, k_don_list, element_type='tri3', is_logging=True):
    """
    Solve the heat transfer problem for a single element.

    Args:
    node (list): A list of Node objects
    mesh (list): A list of Element objects
    T_BC (numpy.ndarray): Boundary condition tensor
    T_nodeid (list): List of node IDs with temperature boundary conditions
    q_BC (numpy.ndarray): Boundary condition tensor for heat flux
    q_nodeid (list): List of node IDs with heat flux boundary conditions
    T_oneBC_nodeid (list): List of node IDs corresponding to the input Temp BC of the trained DON;
                           each element is a list of node IDs
    k (float): Thermal conductivity

    Returns:
    numpy.ndarray: Temperature field
    """
    # if element_type == 'quad4':
    #     import darcy_flow.fem_solver.element.quaelement as quaelement
    #     ele_node_num = 4
    #     element = quaelement
    # elif element_type == 'tri3':
    #     import darcy_flow.fem_solver.element.trielement as element
    #     ele_node_num = 3
    ele_node_num = 4
    num_nodes = len(node)
    appq = np.zeros(num_nodes)
    unbq = np.zeros(num_nodes)
    T = np.zeros(num_nodes)
    for i in range(len(q_nodeid)):
        nodeid = q_nodeid[i]
        appq[nodeid] = q_BC[i]
    unbq += appq
    iter = 0
    mt = np.mean(T_BC)
    if len(T_BC) != 0:
        T_BC = T_BC - mt
        tol_criteria_type = 1
    else:
        tol_criteria_type = 2
    while True:

        row_indices = []
        col_indices = []
        data_values = []

        # Assembly of global stiffness matrix
        for z in range(len(mesh)):
            ele = mesh[z]
            tk = k_fem_list[z]
            ele_coords = np.array([node[i] for i in ele])
            ele_k = element.get_elek(ele_coords, tk)
            for i in range(ele_node_num):
                for j in range(ele_node_num):
                    row_indices.append(ele[i])
                    col_indices.append(ele[j])
                    data_values.append(ele_k[i, j])

        if len(T_oneBC_nodeid_list) != len(k_don_list):
            raise ValueError("The number of T_oneBC_nodeid_list and k_don_list should be the same.")

        for (T_oneBC_nodeid, k_don) in zip(T_oneBC_nodeid_list, k_don_list):
            T_oneBC = np.zeros(len(T_oneBC_nodeid))
            for i in range(len(T_oneBC_nodeid)):
                T_oneBC[i] = T[T_oneBC_nodeid[i]]
            # T_oneBC = T_oneBC - np.mean(T_oneBC)
            K_one = don_element.get_k(T_oneBC, k_don)
            for i in range(len(T_oneBC_nodeid)):
                for j in range(len(T_oneBC_nodeid)):
                    row_indices.append(T_oneBC_nodeid[i])
                    col_indices.append(T_oneBC_nodeid[j])
                    data_values.append(K_one[i, j])
        #
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
            tk = k_fem_list[z]
            ele_coords = np.array([node[i] for i in ele])
            ele_k = element.get_elek(ele_coords, tk)
            ele_T = np.array([T[i] for i in ele])
            ele_q = ele_k @ ele_T
            updated_q[ele] += ele_q

        for (T_oneBC_nodeid, k_don) in zip(T_oneBC_nodeid_list, k_don_list):
            T_oneBC = T[T_oneBC_nodeid]
            q_oneBC = don_element.get_q(T_oneBC, k_don)
            updated_q[T_oneBC_nodeid] += q_oneBC

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
        if iter > 10:
            break
        unbq = delta_q
    return T + mt