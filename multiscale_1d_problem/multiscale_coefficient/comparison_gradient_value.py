import torch
import numpy as np
from meshing import get_don_mesh
import matplotlib.pyplot as plt
import multiscale_1d_problem.solver.fem_solver as fem_solver
num_evluation_points = 20001
def get_don_prediction(U1, U2, x, net):
    x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    x = x % (1/8)
    T_BC_input = torch.tensor([[U1, U2]], dtype=torch.float32)
    branch_input_min = torch.tensor(net.config["branch_input_min"])
    branch_input_max = torch.tensor(net.config["branch_input_max"])
    trunk_input_min = torch.tensor(net.config["trunk_input_min"])
    trunk_input_max = torch.tensor(net.config["trunk_input_max"])
    normalized_branch_input = (T_BC_input - branch_input_min) / (branch_input_max - branch_input_min)
    normalized_trunk_input = (x - trunk_input_min) / (trunk_input_max - trunk_input_min)
    net_output = net.forward_branch_trunk_fixed(normalized_branch_input, normalized_trunk_input)
    output_min = torch.tensor(net.config["output_min"])
    output_max = torch.tensor(net.config["output_max"])
    T = net_output * (output_max - output_min) + output_min
    T = T.view(-1).detach().numpy()
    return T

def get_don_dx_prediction(U1, U2, x, net):
    x0 = torch.tensor(x, dtype=torch.float32, requires_grad=True).view(-1, 1)
    x = x0 % (1/8)
    T_BC_input = torch.tensor([[U1, U2]], dtype=torch.float32)
    branch_input_min = torch.tensor(net.config["branch_input_min"])
    branch_input_max = torch.tensor(net.config["branch_input_max"])
    trunk_input_min = torch.tensor(net.config["trunk_input_min"])
    trunk_input_max = torch.tensor(net.config["trunk_input_max"])
    normalized_branch_input = (T_BC_input - branch_input_min) / (branch_input_max - branch_input_min)
    normalized_trunk_input = (x - trunk_input_min) / (trunk_input_max - trunk_input_min)
    net_output = net.forward_branch_trunk_fixed(normalized_branch_input, normalized_trunk_input)

    output_min = torch.tensor(net.config["output_min"])
    output_max = torch.tensor(net.config["output_max"])
    T = net_output * (output_max - output_min) + output_min
    T_dx = torch.autograd.grad(T, x0, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    return T_dx.view(-1).detach().numpy()

def get_don_prediction_hc(U1, U2, x, net):
    x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    x = x % (1/8)
    T_BC_input = torch.tensor([[U1, U2]], dtype=torch.float32)
    branch_input_min = torch.tensor(net.config["branch_input_min"])
    branch_input_max = torch.tensor(net.config["branch_input_max"])
    trunk_input_min = torch.tensor(net.config["trunk_input_min"])
    trunk_input_max = torch.tensor(net.config["trunk_input_max"])
    normalized_branch_input = (T_BC_input - branch_input_min) / (branch_input_max - branch_input_min)
    normalized_trunk_input = (x - trunk_input_min) / (trunk_input_max - trunk_input_min)
    net_output = net.forward_branch_trunk_fixed(normalized_branch_input, normalized_trunk_input)

    output_min = torch.tensor(net.config["output_min"])
    output_max = torch.tensor(net.config["output_max"])
    T = net_output * (output_max - output_min) + output_min
    y1_end = U1
    y2_end = U2
    T = T * x * (1 / 8 - x) + (y2_end - y1_end) / (1 / 8) * x + y1_end
    T = T.view(-1).detach().numpy()
    return T

def get_don_dx_prediction_hc(U1, U2, x, net):
    x0 = torch.tensor(x, dtype=torch.float32, requires_grad=True).view(-1, 1)
    x = x0 % (1/8)
    T_BC_input = torch.tensor([[U1, U2]], dtype=torch.float32)
    branch_input_min = torch.tensor(net.config["branch_input_min"])
    branch_input_max = torch.tensor(net.config["branch_input_max"])
    trunk_input_min = torch.tensor(net.config["trunk_input_min"])
    trunk_input_max = torch.tensor(net.config["trunk_input_max"])
    normalized_branch_input = (T_BC_input - branch_input_min) / (branch_input_max - branch_input_min)
    normalized_trunk_input = (x - trunk_input_min) / (trunk_input_max - trunk_input_min)
    net_output = net.forward_branch_trunk_fixed(normalized_branch_input, normalized_trunk_input)

    output_min = torch.tensor(net.config["output_min"])
    output_max = torch.tensor(net.config["output_max"])
    T = net_output * (output_max - output_min) + output_min
    y1_end = U1
    y2_end = U2
    T = T * x * (1 / 8 - x) + (y2_end - y1_end) / (1 / 8) * x + y1_end
    T_dx = torch.autograd.grad(T, x0, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_dx = T_dx.view(-1).detach().numpy()
    return T_dx

def is_don(x, segment):
    for i in range(len(segment)):
        if segment[i][0] < x < segment[i][1]:
            return i
    return -1

def single_noe_run(don_segment, U1, U2, net_name, figname=None):
    L = 1
    segment_len = 1 / 8
    ele_num = 500
    node, fem_mesh, don_mesh, k_list, c_list, f_list = get_don_mesh(L, ele_num, don_segment)
    import multiscale_1d_problem.solver.don_solver as don_solver
    import multiscale_1d_problem.solver.element.don_element_ex2test as don_element
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1/8, k=1, c=1)
    U = don_solver.one_solver(node=node, mesh=fem_mesh, U_BC=[U1, U2], U_nodeid=[0, len(node) - 1],
                              f_BC=f_list, f_nodeid=range(len(node)), U_oneBC_nodeid_list=don_mesh,
                              k_list=k_list, c_list=k_list, don_element=don_element, is_logging=True)
    x = np.linspace(0, L, num_evluation_points)
    U_pred = np.zeros_like(x)
    for i in range(len(x)):
        don_segment_idx = is_don(x[i], don_segment)
        if don_segment_idx != -1:
            mesh = don_mesh[don_segment_idx]
            tU_pred = get_don_prediction(U[mesh[0]], U[mesh[1]], x[i], don_element.DON_info.net)[0]
            U_pred[i] = tU_pred
        else:
            tU_pred = np.interp(x[i], node, U)
            U_pred[i] = tU_pred

    x_fem = []
    U_fem = []
    x_don = []
    U_don = []

    for mesh in fem_mesh:
        x1 = node[mesh[0]]
        x2 = node[mesh[1]]
        x_fem.append([x1, x2])
        U_fem.append([U[mesh[0]], U[mesh[1]]])

    for mesh in don_mesh:
        tx = np.linspace(0.001 * segment_len, 0.999 * segment_len, 51) + node[mesh[0]]
        T = get_don_prediction(U[mesh[0]], U[mesh[1]], tx, don_element.DON_info.net)
        x_don.append(tx)
        U_don.append(T)

    return x, U_pred, x_fem, U_fem, x_don, U_don

def single_noe_runv2(don_segment, U1, U2, net_name, figname=None):
    L = 1
    node = np.linspace(0, L, 9)
    don_mesh = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
    fem_mesh = []
    f_list = []
    k_list = []
    c_list = []
    import multiscale_1d_problem.solver.don_solver as don_solver
    import multiscale_1d_problem.solver.element.don_element as don_element
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1/8, k=1, c=1)
    U = don_solver.one_solver(node=node, mesh=fem_mesh, U_BC=[U1, U2], U_nodeid=[0, len(node) - 1],
                              f_BC=f_list, f_nodeid=[], U_oneBC_nodeid_list=don_mesh,
                              k_list=k_list, c_list=c_list, don_element=don_element, is_logging=True)
    x = np.linspace(0, L, num_evluation_points)
    U_dx_pred = np.zeros_like(x)
    for i in range(len(x)):
        don_segment_idx = is_don(x[i], don_segment)
        if don_segment_idx != -1:
            mesh = don_mesh[don_segment_idx]
            tU_pred = get_don_dx_prediction(U[mesh[0]], U[mesh[1]], x[i], don_element.DON_info.net)[0]
            U_dx_pred[i] = tU_pred
        else:
            tU_pred = np.interp(x[i], node, U)
            U_dx_pred[i] = tU_pred

    x_fem = []
    U_fem = []
    x_don = []
    U_don = []
    U_dx_don = []

    for mesh in fem_mesh:
        x1 = node[mesh[0]]
        x2 = node[mesh[1]]
        x_fem.append([x1, x2])
        U_fem.append([U[mesh[0]], U[mesh[1]]])
    segment_len = 1 / 8
    for mesh in don_mesh:
        tx = np.linspace(0.001 * segment_len, 0.999 * segment_len, 51) + node[mesh[0]]
        T = get_don_prediction(U[mesh[0]], U[mesh[1]], tx, don_element.DON_info.net)
        T_dx = get_don_dx_prediction(U[mesh[0]], U[mesh[1]], tx, don_element.DON_info.net)
        x_don.append(tx)
        U_don.append(T)
        U_dx_don.append(T_dx)

    return x, U_dx_pred, x_fem, U_fem, x_don, U_don, U_dx_don

def single_noe_runv2hc(don_segment, U1, U2, net_name, figname=None):
    L = 1
    node = np.linspace(0, L, 9)
    don_mesh = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
    fem_mesh = []
    f_list = []
    k_list = []
    c_list = []
    import multiscale_1d_problem.solver.don_solver as don_solver
    import multiscale_1d_problem.solver.element.don_element_hc as don_element
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1/8, k=1, c=1)
    U = don_solver.one_solver(node=node, mesh=fem_mesh, U_BC=[U1, U2], U_nodeid=[0, len(node) - 1],
                              f_BC=f_list, f_nodeid=[], U_oneBC_nodeid_list=don_mesh,
                              k_list=k_list, c_list=c_list, don_element=don_element, is_logging=True)
    x = np.linspace(0, L, num_evluation_points)
    U_pred = np.zeros_like(x)
    for i in range(len(x)):
        don_segment_idx = is_don(x[i], don_segment)
        if don_segment_idx != -1:
            mesh = don_mesh[don_segment_idx]
            tU_pred = get_don_dx_prediction_hc(U[mesh[0]], U[mesh[1]], x[i], don_element.DON_info.net)[0]
            U_pred[i] = tU_pred
        else:
            tU_pred = np.interp(x[i], node, U)
            U_pred[i] = tU_pred

    x_fem = []
    U_fem = []
    x_don = []
    U_don = []
    U_dx_don = []

    for mesh in fem_mesh:
        x1 = node[mesh[0]]
        x2 = node[mesh[1]]
        x_fem.append([x1, x2])
        U_fem.append([U[mesh[0]], U[mesh[1]]])
    segment_len = 1 / 8
    for mesh in don_mesh:
        tx = np.linspace(0.001 * segment_len, 0.999 * segment_len, 51) + node[mesh[0]]
        T = get_don_prediction_hc(U[mesh[0]], U[mesh[1]], tx, don_element.DON_info.net)
        T_dx = get_don_dx_prediction_hc(U[mesh[0]], U[mesh[1]], tx, don_element.DON_info.net)
        x_don.append(tx)
        U_don.append(T)
        U_dx_don.append(T_dx)

    return x, U_pred, x_fem, U_fem, x_don, U_don, U_dx_don

def single_fem_run(U1, U2):
    L = 1
    node_num = num_evluation_points
    node = np.linspace(0, L, node_num)
    U_BC = [U1, U2]
    U_nodeid = [0, node_num - 1]
    f_BC = []
    f_nodeid = []
    for i in range(len(node)):
        x = node[i]
        f_nodeid.append(i)
        x = x % 1
        print(x)
        f_BC.append(0.5 * L / (node_num - 1))

    mesh = [[i, i + 1] for i in range(node_num - 1)]
    c_list = np.zeros(len(mesh)).tolist()
    ele_loc = [0.5 * (node[i] + node[i + 1]) for i in range(node_num - 1)]
    ele_loc = np.array(ele_loc)
    wave_num = 16
    k_list = 0.5 * np.sin(2 * np.pi * ele_loc * wave_num) + 0.8
    T = fem_solver.solver(node, mesh, U_BC, U_nodeid, f_BC, f_nodeid, k_list, c_list)
    return node, T

def get_fem_dx(x_fem, U_fem):
    dx = x_fem[1] - x_fem[0]
    U_dx_fem = np.array(U_fem[1:] - U_fem[:-1]) / dx
    U_dx_x0 = (U_fem[1] - U_fem[0]) / dx
    U_dx_fem = np.concatenate(([U_dx_x0], U_dx_fem))
    return U_dx_fem

def run():
    U1 = 0.
    U2 = 0
    x_fem, U_fem = single_fem_run(U1, U2)
    U_dx_fem = get_fem_dx(x_fem, U_fem)
    # Creat two subplots
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(4,4))
    axs = axs.flatten()
    axs[0].plot(x_fem, U_dx_fem, "black")
    don_segment_list = [[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]]
    don_segment_list = np.array(don_segment_list) / 8
    don_segment_list = don_segment_list.tolist()

    for i, don_segment in enumerate(don_segment_list):
        net_name = r"deeponet.pth"
        print(don_segment)
        x_don, U_dx_don, noe_x_fem, noe_U_fem, noe_x_don, noe_U_don, noe_U_dx_don = single_noe_runv2(don_segment, U1, U2, net_name)
        diff = U_dx_don - U_dx_fem
        idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        idx_list = np.array(idx_list) * int((num_evluation_points - 1) / 8)
        idx_list = idx_list.tolist()
        diff[idx_list] = 0
        relative_L2_error = np.linalg.norm(diff) / np.linalg.norm(U_dx_fem)
        for j in range(len(noe_x_fem)):
            axs[1].plot(noe_x_fem[j], noe_U_fem[j], "blue")
        for j in range(len(noe_x_don)):
            axs[1].plot(noe_x_don[j], noe_U_dx_don[j], "red")


    for i, don_segment in enumerate(don_segment_list):
        net_name = r"hc_deeponet.pth"
        print(don_segment)
        x_don, U_dx_don, noe_x_fem, noe_U_fem, noe_x_don, noe_U_don, noe_U_dx_don = single_noe_runv2hc(don_segment, U1, U2, net_name)
        diff = U_dx_don - U_dx_fem
        idx_list = [0,1,2,3,4,5,6,7,8]
        idx_list = np.array(idx_list) * int((num_evluation_points - 1) / 8)
        idx_list = idx_list.tolist()
        diff[idx_list] = 0
        relative_L2_error = np.linalg.norm(diff) / np.linalg.norm(U_dx_fem)
        for j in range(len(noe_x_fem)):
            axs[i + 2].plot(noe_x_fem[j], noe_U_fem[j], "blue")
        for j in range(len(noe_x_don)):
            axs[i + 2].plot(noe_x_don[j], noe_U_dx_don[j], "orange")
    plt.tight_layout()
    plt.savefig("8_noe_segment.png")
    plt.show()

if __name__ == '__main__':
    pass
