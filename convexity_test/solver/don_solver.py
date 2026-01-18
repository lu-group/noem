from ex2.solver.element import line_element as element
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
def one_solver(node, mesh, U_BC, U_nodeid, f_BC, f_nodeid,
               U_oneBC_nodeid_list, k_list, c_list, don_element,
               is_logging=False, u_init=None):
    ele_node_num = 2
    num_nodes = len(node)

    appF = np.zeros(num_nodes)
    appF[f_nodeid] = f_BC

    for i in range(len(U_nodeid)):
        appF[U_nodeid[i]] = 0.0

    unbF = appF.copy()

    U = np.zeros(num_nodes)
    if u_init is not None:
        U = u_init.copy()

    bc_dofs = np.array(U_nodeid, dtype=int)
    U_BC_arr = np.array(U_BC, dtype=float)

    all_dofs = np.arange(num_nodes, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, bc_dofs)

    U[bc_dofs] = U_BC_arr

    iter = 0
    while True:
        U[bc_dofs] = U_BC_arr

        row_indices = []
        col_indices = []
        data_values = []

        for k, c, ele in zip(k_list, c_list, mesh):
            ele_coords = np.array([node[i] for i in ele])
            ele_k = element.get_elek(ele_coords, k, c)
            for i_loc in range(ele_node_num):
                for j_loc in range(ele_node_num):
                    row_indices.append(ele[i_loc])
                    col_indices.append(ele[j_loc])
                    data_values.append(ele_k[i_loc, j_loc])

        for U_oneBC_nodeid in U_oneBC_nodeid_list:
            U_oneBC = U[U_oneBC_nodeid]
            K_one = don_element.get_k(U_oneBC)
            for i_loc in range(len(U_oneBC_nodeid)):
                for j_loc in range(len(U_oneBC_nodeid)):
                    row_indices.append(U_oneBC_nodeid[i_loc])
                    col_indices.append(U_oneBC_nodeid[j_loc])
                    data_values.append(K_one[i_loc, j_loc])

        K = csr_matrix((data_values, (row_indices, col_indices)),
                       shape=(num_nodes, num_nodes))

        equal_unbF = unbF.copy()

        if free_dofs.size == 0:
            U[bc_dofs] = U_BC_arr
            return U

        K_ff = K[free_dofs, :][:, free_dofs]
        rhs_ff = equal_unbF[free_dofs]

        delta_U = np.zeros(num_nodes)
        delta_U_ff = spsolve(K_ff, rhs_ff)
        delta_U[free_dofs] = delta_U_ff

        U += delta_U
        U[bc_dofs] = U_BC_arr

        updated_F = np.zeros(num_nodes)

        for k, c, ele in zip(k_list, c_list, mesh):
            ele_coords = np.array([node[i] for i in ele])
            ele_k = element.get_elek(ele_coords, k, c)
            ele_U = np.array([U[i] for i in ele])
            ele_F = ele_k @ ele_U
            updated_F[ele] += ele_F

        for U_oneBC_nodeid in U_oneBC_nodeid_list:
            U_oneBC = U[U_oneBC_nodeid]
            F_oneBC = don_element.get_q(U_oneBC)
            updated_F[U_oneBC_nodeid] += F_oneBC

        updated_F[bc_dofs] = 0.0

        delta_F = appF - updated_F

        num = np.linalg.norm(delta_U[free_dofs])
        den = np.linalg.norm(U[free_dofs])
        if den < 1e-16:
            tol_criteria = num
        else:
            tol_criteria = num / den

        if is_logging:
            print("magnitude of delta_T:", tol_criteria)

        if tol_criteria < 1e-3:
            break

        iter += 1
        if iter > 10:
            return None

        unbF = delta_F

    return U

def convergence_test(u2, u3):
    node = np.linspace(0, 3, 4)
    fem_mesh = [[0, 1], [2, 3]]
    don_mesh = [[1, 2]]
    U1 = 0
    u_init = [0, u2, u3, 0]
    u_init = np.array(u_init)
    from convexity_test.solver.element import don_element as don_element
    net_name = r"C:\Users\Weihang Ouyang\Desktop\NOEM Code\convexity_test\ex_new\deeponet.pth"
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1)
    try:
        U = one_solver(node=node, mesh=fem_mesh, U_BC=[U1], U_nodeid=[0], f_BC=[0.1], f_nodeid=[3],
                       U_oneBC_nodeid_list=don_mesh, k_list=[1, 1], c_list=[0, 0], don_element=don_element,
                       is_logging=True, u_init=u_init)
    except:
        U = None
    if U is None:
        print("Did not converge")
    print(U)
    if U is None:
        return False
    return True
if __name__ == '__main__':
    node = np.linspace(0, 3, 4)
    fem_mesh = [[0, 1], [2,3]]
    don_mesh = [[1,2]]
    U1 = 0
    u2, u3 = [2,0]
    u_init = [0, u2, u3, 0]
    u_init = np.array(u_init)
    from convexity_test.solver.element import don_element as don_element
    net_name = r"C:\Users\Weihang Ouyang\Desktop\NOEM Code\convexity_test\ex_new\deeponet.pth"
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1)
    try:
        U = one_solver(node=node, mesh=fem_mesh, U_BC=[U1], U_nodeid=[0], f_BC=[0.1], f_nodeid=[3], U_oneBC_nodeid_list=don_mesh, k_list=[1, 1], c_list=[0, 0], don_element=don_element, is_logging=True, u_init=u_init)
    except:
        U = None
    if U is None:
        print("Did not converge")
    print(U)