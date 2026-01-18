from ex3.fem_solver.element import trielement as trielement
# from heat_transfer.fem_solver.element import quaelement as quaelement
import numpy as np
import torch
def one_solver(node, mesh, T_BC, T_nodeid, q_BC, q_nodeid, T_oneBC_nodeid, don_element, k=1, element_type='tri3'):
    """
    Solve the heat transfer problem for a single element.

    Args:
    node (list): A list of Node objects
    mesh (list): A list of Element objects
    T_BC (numpy.ndarray): Boundary condition tensor
    T_nodeid (list): List of node IDs with temperature boundary conditions
    q_BC (numpy.ndarray): Boundary condition tensor for heat flux
    q_nodeid (list): List of node IDs with heat flux boundary conditions
    T_oneBC_nodeid (list): List of node IDs corresponding to the input Temp BC of the trained DON
    k (float): Thermal conductivity

    Returns:
    numpy.ndarray: Temperature field
    """
    if element_type == 'tri3':
        elenum_dof_num = 3
        element = trielement
    elif element_type == 'quad4':
        elenum_dof_num = 4
        element = quaelement

    num_nodes = len(node)
    appq = np.zeros(num_nodes)
    unbq = np.zeros(num_nodes)
    T = np.zeros(num_nodes)
    for i in range(len(q_nodeid)):
        nodeid = q_nodeid[i]
        appq[nodeid] = q_BC[i]
    unbq += appq
    iter = 0
    while True:
        # Initialize the global stiffness matrix and load vector
        K = np.zeros((num_nodes, num_nodes))

        # Assemble the global stiffness matrix and load vector
        for ele in mesh:
            ele_coords = np.array([node[i] for i in ele])
            ele_k = element.get_elek(ele_coords, k)  # Assuming the function is from a module named element
            for i in range(elenum_dof_num):
                for j in range(elenum_dof_num):
                    K[ele[i], ele[j]] += ele_k[i, j]

        T_oneBC = np.zeros(len(T_oneBC_nodeid))
        for i in range(len(T_oneBC_nodeid)):
            T_oneBC[i] = T[T_oneBC_nodeid[i]]

        K_one = don_element.get_k(T_oneBC)
        # Assemble K_one to K according to T_oneBC_nodeid
        for i in range(len(T_oneBC_nodeid)):
            for j in range(len(T_oneBC_nodeid)):
                K[T_oneBC_nodeid[i], T_oneBC_nodeid[j]] += K_one[i, j]

        equal_unbq = unbq
        # Apply temperature boundary conditions
        average_k = np.mean(np.abs(np.diag(K)))
        beta = 1e7 * average_k
        for idx, Ti in zip(T_nodeid, T_BC):
            K[idx, idx] = beta
            if iter == 0:
                equal_unbq[idx] = Ti * beta
            else:
                equal_unbq[idx] = 0

        delta_T = np.linalg.solve(K, equal_unbq)
        T += delta_T

        updated_q = np.zeros(num_nodes)
        # Solve the system of equations
        for ele in mesh:
            ele_coords = np.array([node[i] for i in ele])
            ele_k = element.get_elek(ele_coords, k)  # Assuming the function is from a module named element
            ele_T = np.array([T[i] for i in ele])
            ele_q = ele_k @ ele_T
            for i in range(elenum_dof_num):
                updated_q[ele[i]] += ele_q[i]

        T_oneBC = np.zeros(len(T_oneBC_nodeid))
        # Assemble q_oneBC to updated_q
        for i in range(len(T_oneBC_nodeid)):
            T_oneBC[i] = T[T_oneBC_nodeid[i]]

        q_oneBC = don_element.get_q(T_oneBC)
        for i in range(len(T_oneBC_nodeid)):
            updated_q[T_oneBC_nodeid[i]] += q_oneBC[i]

        # apply BC
        for i in range(len(T_nodeid)):
            updated_q[T_nodeid[i]] = 0

        delta_q = appq - updated_q
        print("magnitude of delta_q:", np.linalg.norm(delta_q))
        if np.linalg.norm(delta_q) < 1e-5:
            break

        iter += 1
        if iter > 5:
            break
        unbq = delta_q
    return T