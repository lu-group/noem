# This modulus supports the analysis domain containing multiple don elements.

from ex2.solver.element import line_element as element
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import torch, time
def one_solver(node, mesh, U_BC, U_nodeid, f_BC, f_nodeid, U_oneBC_nodeid_list, U_one_para_list, k_list, c_list, don_element, is_logging=False):
    ele_node_num = 2
    num_nodes = len(node)
    appF = np.zeros(num_nodes)
    appF[f_nodeid] = f_BC
    for i in range(len(U_nodeid)):
        appF[U_nodeid[i]] = 0
    unbF = np.zeros(num_nodes)
    unbF += appF
    U = np.zeros(num_nodes)
    iter = 0
    while True:
        row_indices = []
        col_indices = []
        data_values = []

        # Assemble the global stiffness matrix and load vector
        for k, c, ele in zip(k_list, c_list, mesh):
            ele_coords = np.array([node[i] for i in ele])
            ele_k = element.get_elek(ele_coords, k, c)
            for i in range(ele_node_num):
                for j in range(ele_node_num):
                    row_indices.append(ele[i])
                    col_indices.append(ele[j])
                    data_values.append(ele_k[i, j])

        for U_oneBC_nodeid, U_one_para in zip(U_oneBC_nodeid_list, U_one_para_list):
            U_oneBC = U[U_oneBC_nodeid]
            wave_num, x0 = U_one_para
            K_one = don_element.get_k(U_oneBC, wave_num, x0)
            for i in range(len(U_oneBC_nodeid)):
                for j in range(len(U_oneBC_nodeid)):
                    row_indices.append(U_oneBC_nodeid[i])
                    col_indices.append(U_oneBC_nodeid[j])
                    data_values.append(K_one[i, j])

        K = csr_matrix((data_values, (row_indices, col_indices)), shape=(len(node), len(node)))
        equal_unbF = unbF
        # Apply temperature boundary conditions
        average_k = np.mean(np.max(K.diagonal()))
        beta = 1e7 * average_k
        big_indices = np.array(U_nodeid, dtype=int)
        K[big_indices, big_indices] = beta
        if iter == 0:
            equal_unbF[big_indices] = beta * np.array(U_BC)
        else:
            equal_unbF[big_indices] = 0

        delta_U = spsolve(K, equal_unbF)
        U += delta_U

        updated_F = np.zeros(num_nodes)
        # Solve the system of equations
        for k, c, ele in zip(k_list, c_list, mesh):
            ele_coords = np.array([node[i] for i in ele])
            ele_k = element.get_elek(ele_coords, k, c)
            ele_U = np.array([U[i] for i in ele])
            ele_F = ele_k @ ele_U
            updated_F[ele] += ele_F

        for U_oneBC_nodeid, U_one_para in zip(U_oneBC_nodeid_list, U_one_para_list):
            U_oneBC = U[U_oneBC_nodeid]
            wave_num, x0 = U_one_para
            F_oneBC = don_element.get_q(U_oneBC, wave_num, x0)
            updated_F[U_oneBC_nodeid] += F_oneBC
        # apply BC
        for i in range(len(U_nodeid)):
            updated_F[U_nodeid[i]] = 0

        delta_F = appF - updated_F

        tol_criteria = np.linalg.norm(delta_U) / np.linalg.norm(U)
        if is_logging:
            print("magnitude of delta_T:", tol_criteria)
        if tol_criteria < 1e-3:
            break

        iter += 1
        if iter > 10:
            break
        unbF = delta_F
    return U