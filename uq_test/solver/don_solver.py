# This modulus supports the analysis domain containing multiple don elements.

from ex2.solver.element import line_element as element
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

import torch


class UQ_Evaluator:
    net_name = r"C:\Users\Weihang Ouyang\Desktop\NOEM Code\uq_test\ex_new\models\L=1_samnum=5000_mionet"

    @staticmethod
    def cal_uq_score(u_bc_input, k_dis):
        uq_deeponet_result_list = []
        for j in range(10):
            tnet_name = UQ_Evaluator.net_name + "_rep=" + str(j) + ".pth"
            tno_pred = UQ_Evaluator.single_no_run(
                net=torch.load(tnet_name, map_location=torch.device('cpu')),
                U_bc=u_bc_input,
                k_dis=k_dis,
                trunk_input=np.linspace(0, 1, 51)
            )
            uq_deeponet_result_list.append(tno_pred)
        uq_deeponet_result_array = np.array(uq_deeponet_result_list)
        # Compute the variance along axis 0
        variance = np.var(uq_deeponet_result_array, axis=0)
        std = np.std(uq_deeponet_result_array, axis=0)
        # mean = np.mean(np.abs(uq_deeponet_result_array), axis=0)
        # Compute the UQ score as the mean of variance divided by mean
        uq_score = np.mean(std)
        return uq_score

    @staticmethod
    def single_no_run(net, U_bc, k_dis, trunk_input):
        device = "cpu"
        sampling_points = trunk_input.reshape(-1, 1)
        sampling_points = torch.tensor(sampling_points, device=device, dtype=torch.float32)
        T_BC = torch.tensor([U_bc[0], U_bc[1]], dtype=torch.float32).view(1, -1)
        k_dist = torch.tensor(k_dis, dtype=torch.float32).view(1, -1)
        branch_input1_min = torch.tensor(net.config["branch_input1_min"])
        branch_input1_max = torch.tensor(net.config["branch_input1_max"])
        branch_input2_min = torch.tensor(net.config["branch_input2_min"])
        branch_input2_max = torch.tensor(net.config["branch_input2_max"])
        trunk_input_min = torch.tensor(net.config["trunk_input_min"])
        trunk_input_max = torch.tensor(net.config["trunk_input_max"])
        normalized_branch_input1 = (T_BC - branch_input1_min) / (branch_input1_max - branch_input1_min)
        normalized_branch_input2 = (k_dist - branch_input2_min) / (branch_input2_max - branch_input2_min)
        normalized_trunk_input = (sampling_points - trunk_input_min) / (trunk_input_max - trunk_input_min)

        net_output = net.forward_branch_trunk_fixed(
            branch_input_list=[normalized_branch_input1, normalized_branch_input2],
            trunk_input=normalized_trunk_input)
        output_min = torch.tensor(net.config["output_min"])
        output_max = torch.tensor(net.config["output_max"])
        net_output = net_output * (output_max - output_min) + output_min

        return net_output.view(-1).detach().numpy().tolist()

def one_solver(node, mesh, U_BC, U_nodeid, f_BC, f_nodeid, U_oneBC_nodeid_list, k_list, c_list, don_element, U_one_finput_list, is_logging=False):
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
    Score_list = []
    while True:
        row_indices = []
        col_indices = []
        data_values = []
        energy = 0
        # Assemble the global stiffness matrix and load vector
        for k, c, ele in zip(k_list, c_list, mesh):
            ele_coords = np.array([node[i] for i in ele])
            ele_k = element.get_elek(ele_coords, k, c)
            # Compute the energy at this element
            ele_U = np.array([U[i] for i in ele])
            ele_len = ele_coords[1] - ele_coords[0]
            ele_energy_density = 0.5 * k * ((ele_U[1] - ele_U[0]) / ele_len) ** 2
            ele_energy = ele_energy_density * ele_len
            energy += ele_energy
            for i in range(ele_node_num):
                for j in range(ele_node_num):
                    row_indices.append(ele[i])
                    col_indices.append(ele[j])
                    data_values.append(ele_k[i, j])

        for U_oneBC_nodeid, U_one_finput in zip(U_oneBC_nodeid_list, U_one_finput_list):
            U_oneBC = U[U_oneBC_nodeid]
            K_one, noe_energy = don_element.get_k(U_oneBC, U_one_finput)
            energy += noe_energy
            uq_score = UQ_Evaluator.cal_uq_score(U_oneBC, U_one_finput)
            Score_list.append(uq_score)
            for i in range(len(U_oneBC_nodeid)):
                for j in range(len(U_oneBC_nodeid)):
                    row_indices.append(U_oneBC_nodeid[i])
                    col_indices.append(U_oneBC_nodeid[j])
                    data_values.append(K_one[i, j])

        K = csr_matrix((data_values, (row_indices, col_indices)), shape=(len(node), len(node)))
        equal_unbF = unbF
        # Add energy from applied forces
        for i in f_nodeid:
            ele_energy = U[i] * f_BC[f_nodeid.index(i)]
            energy -= ele_energy
        # print(iter, energy)
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

        for U_oneBC_nodeid, U_one_finput in zip(U_oneBC_nodeid_list, U_one_finput_list):
            U_oneBC = U[U_oneBC_nodeid]
            F_oneBC, noe_energy = don_element.get_q(U_oneBC, U_one_finput)
            updated_F[U_oneBC_nodeid] += F_oneBC
        # apply BC
        for i in range(len(U_nodeid)):
            updated_F[U_nodeid[i]] = 0

        delta_F = appF - updated_F

        tol_criteria = np.linalg.norm(delta_U) / np.linalg.norm(U)
        if is_logging:
            print("magnitude of delta_T:", tol_criteria)
        if tol_criteria < 1e-3: # and iter > -1:
            break

        iter += 1
        if iter > 10:
            return None, Score_list
        unbF = delta_F

    return U,Score_list