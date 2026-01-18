import torch
import numpy as np
import uq_test.solver.don_solver as don_solver
import uq_test.solver.element.don_elementnew as don_element
import pandas as pd

class Error_Metric:
    type = "L2"

    @staticmethod
    def cal(error, reference):
        if Error_Metric.type == "L2":
            return np.linalg.norm(error) / np.linalg.norm(reference)
        elif Error_Metric.type == "MeanL2":
            relative_L2_errors = np.linalg.norm(error, axis=1) / np.linalg.norm(reference, axis=1)
            average_relative_L2_error = np.mean(relative_L2_errors)
            return average_relative_L2_error

def single_noe_run(net_name, U_bc, k_dis):
    node1 = np.linspace(0, 1, 51)
    node2 = node1 + 2
    node = np.concatenate([node1, node2])
    fem_mesh1 = [[i, i + 1] for i in range(50)]
    fem_mesh2 = [[i + 51, i + 52] for i in range(50)]
    fem_mesh = fem_mesh1 + fem_mesh2
    don_mesh = [[50, 51]]
    U1, U2 = U_bc
    U_one_finput_list = np.array(k_dis)
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1)
    U, uq_score_list = don_solver.one_solver(node=node, mesh=fem_mesh, U_BC=[0], U_nodeid=[0], f_BC=[0.05], f_nodeid=[len(node) - 1],
                              U_oneBC_nodeid_list=don_mesh, U_one_finput_list=U_one_finput_list,
                              k_list=[1 for i in range(len(fem_mesh))], c_list=[0 for i in range(len(fem_mesh))],
                              don_element=don_element, is_logging=False)
    return U, uq_score_list

def recover_sol(net_name, U, k_dis):
    net = torch.load(net_name, map_location=torch.device('cpu'))
    output_results = []
    fem_u1 = U[:51]
    fem_u2 = U[51:102]
    U1, U2 = U[50], U[51]
    k_input = k_dis

    num_gauss = 49
    sampling_points = [(i + 1) / (num_gauss + 1) for i in range(num_gauss)]
    device = "cpu"
    x_len = 1
    sampling_points = torch.tensor(sampling_points, device=device, dtype=torch.float32, requires_grad=True) * x_len
    sampling_points = sampling_points.view(-1, 1)
    T_BC = torch.tensor([U1, U2], dtype=torch.float32).view(1, -1)
    k_dist = torch.tensor(k_input, dtype=torch.float32).view(1, -1)
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

    noe_results = net_output.view(-1).detach().numpy().tolist()
    sol = fem_u1.tolist() + noe_results + fem_u2.tolist()
    return sol

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

def single_fem_run_L_1(u_bc, k_dis):
    node = np.linspace(0, 1, 51)
    fem_mesh = [[i, i + 1] for i in range(50)]
    import uq_test.solver.fem_solver as fem_solver
    U = fem_solver.solver(node=node, mesh=fem_mesh, U_BC=u_bc, U_nodeid=[0, 50], f_BC=[], f_nodeid=[],
                          k_list=k_dis, c_list=[0 for i in range(len(fem_mesh))])
    return U

def single_fem_run_L_3(k_dis, U_bc):
    node1 = np.linspace(0, 1, 51)
    node2 = node1[1:] + 1
    node3 = node1[1:] + 2
    node = np.concatenate([node1, node2, node3])
    fem_mesh = [[i, i + 1] for i in range(150)]
    U1 = 0
    f1 = 0.1
    k_list = [1 for i in range(len(fem_mesh))]
    # replace the middle part of k_list with k_dis
    k_list[50:100] = k_dis
    import uq_test.solver.fem_solver as fem_solver
    # U = fem_solver.solver(node=node, mesh=fem_mesh, U_BC=U_bc, U_nodeid=[0,len(node) - 1], f_BC=[], f_nodeid=[],
    #                       k_list=k_list, c_list=[0 for i in range(len(fem_mesh))])
    U = fem_solver.solver(node=node, mesh=fem_mesh, U_BC=[0], U_nodeid=[0], f_BC=[0.05], f_nodeid=[len(node) - 1],
                          k_list=k_list, c_list=[0 for i in range(len(fem_mesh))])
    return U

def run(sam_num, l=0.3):

    from util.grf import batch_grf_1d
    ele_loc = [(0.5 + i) / 50 for i in range(50)]
    k_dis_list = batch_grf_1d(np.array(ele_loc), l=l, batch_size=sam_num, sigma=1.0, jitter=1e-8)
    k_dis_list = np.exp(k_dis_list)
    net_name = r"ex_new\models\L=1_samnum=5000_mionet"
    result_list = []
    uq_score_list = []
    for i in range(sam_num):
        k_dis = k_dis_list[i]
        U_bc = [0, 0.5 + 0.1 * i]
        fem_sol = single_fem_run_L_3(k_dis=k_dis, U_bc=U_bc)
        tnet_name = net_name + ".pth"
        results, uq_score = single_noe_run(net_name=tnet_name, k_dis=k_dis, U_bc=U_bc)
        result_list.append(results)
        uq_score_list.append(uq_score)
    converged = []
    diverged = []
    for i in range(len(result_list)):
        if result_list[i] is None:
            diverged.append([uq_score_list[i][-1]])
        else:
            converged.append([uq_score_list[i][-1]])

    print("="*50)
    print("Converged UQ scores:")
    print(np.array(converged).reshape(-1).tolist())
    print("Diverged UQ scores:")
    for i in diverged:
        print(i[0])

    print("=" * 50)
    print("Average results:")
    converged = []
    diverged = []
    for i in range(len(result_list)):
        meanvalue = np.mean(uq_score_list[i])
        if result_list[i] is None:
            diverged.append(meanvalue)
        else:
            converged.append(meanvalue)
    print("Converged UQ scores:")
    print(np.array(converged).reshape(-1).tolist())
    print("Diverged UQ scores:")
    for i in diverged:
        print(i)








if __name__ == "__main__":
    # Fix random seed
    noem_accu_list = []
    no_accu_list = []
    uq_score_list = []
    l_list = [0.1]
    noem_accu_std_list = []
    no_accu_std_list = []
    uq_score_std_list = []
    for l in l_list:
        print("Running for l =", l)
        run(sam_num=1000, l=l)

