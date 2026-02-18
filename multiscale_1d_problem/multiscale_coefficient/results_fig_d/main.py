import torch
import numpy as np
import matplotlib.pyplot as plt
import ex2.solver.fem_solver as fem_solver

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
    # input_tensor = torch.cat((normalized_branch_input, normalized_trunk_input), dim=1)
    # net_output = net(input_tensor)
    net_output = net.forward_branch_trunk_fixed(normalized_branch_input, normalized_trunk_input)

    output_min = torch.tensor(net.config["output_min"])
    output_max = torch.tensor(net.config["output_max"])
    T = net_output * (output_max - output_min) + output_min
    y1_end = U1
    y2_end = U2
    T = T * x * (1 / 8 - x) + (y2_end - y1_end) / (1 / 8) * x + y1_end
    T = T.view(-1).detach().numpy()
    return T

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
    import ex2.solver.don_solver as don_solver
    import ex2.solver.element.don_element_ex2test as don_element
    # net_name = r"..\\..\\data_driven_training/oned_ell/ode_sample10000_100meshv2.pth"
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1/8, k=1, c=1)
    U = don_solver.one_solver(node=node, mesh=fem_mesh, U_BC=[U1, U2], U_nodeid=[0, len(node) - 1],
                              f_BC=f_list, f_nodeid=range(len(node)), U_oneBC_nodeid_list=don_mesh,
                              k_list=k_list, c_list=k_list, don_element=don_element, is_logging=True)
    x = np.linspace(0, L, 1001)
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
        tx = np.linspace(0.01 * segment_len, 0.99 * segment_len, 51) + node[mesh[0]]
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
    import ex2.solver.don_solver as don_solver
    import ex2.solver.element.don_element as don_element
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1/8, k=1, c=1)
    U = don_solver.one_solver(node=node, mesh=fem_mesh, U_BC=[U1, U2], U_nodeid=[0, len(node) - 1],
                              f_BC=f_list, f_nodeid=[], U_oneBC_nodeid_list=don_mesh,
                              k_list=k_list, c_list=c_list, don_element=don_element, is_logging=False)
    x = np.linspace(0, L, 1001)
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
    segment_len = 1 / 8
    for mesh in don_mesh:
        tx = np.linspace(0.01 * segment_len, 0.99 * segment_len, 51) + node[mesh[0]]
        T = get_don_prediction(U[mesh[0]], U[mesh[1]], tx, don_element.DON_info.net)
        x_don.append(tx)
        U_don.append(T)

    return x, U_pred, x_fem, U_fem, x_don, U_don

def single_noe_runv2hc(don_segment, U1, U2, net_name, figname=None):
    L = 1
    node = np.linspace(0, L, 9)
    don_mesh = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
    fem_mesh = []
    f_list = []
    k_list = []
    c_list = []
    import ex2.solver.don_solver as don_solver
    import ex2.solver.element.don_element_ex2testv2_hc as don_element
    # net_name = r"..\\..\\data_driven_training/oned_ell/ode_sample10000_100meshv2.pth"
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1/8, k=1, c=1)
    U = don_solver.one_solver(node=node, mesh=fem_mesh, U_BC=[U1, U2], U_nodeid=[0, len(node) - 1],
                              f_BC=f_list, f_nodeid=[], U_oneBC_nodeid_list=don_mesh,
                              k_list=k_list, c_list=c_list, don_element=don_element, is_logging=False)
    x = np.linspace(0, L, 1001)
    U_pred = np.zeros_like(x)
    for i in range(len(x)):
        don_segment_idx = is_don(x[i], don_segment)
        if don_segment_idx != -1:
            mesh = don_mesh[don_segment_idx]
            tU_pred = get_don_prediction_hc(U[mesh[0]], U[mesh[1]], x[i], don_element.DON_info.net)[0]
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
    segment_len = 1 / 8
    for mesh in don_mesh:
        tx = np.linspace(0.01 * segment_len, 0.99 * segment_len, 51) + node[mesh[0]]
        T = get_don_prediction_hc(U[mesh[0]], U[mesh[1]], tx, don_element.DON_info.net)
        x_don.append(tx)
        U_don.append(T)

    return x, U_pred, x_fem, U_fem, x_don, U_don

def single_fem_run(U1, U2):
    L = 1
    node_num = 1001
    node = np.linspace(0, L, node_num)
    U_BC = [U1, U2]
    U_nodeid = [0, node_num - 1]
    f_BC = []
    f_nodeid = []
    for i in range(len(node)):
        x = node[i]
        f_nodeid.append(i)
        x = x % 1
        # print(x)
        f_BC.append(0.5 * L / (node_num - 1))

    mesh = [[i, i + 1] for i in range(node_num - 1)]
    c_list = np.zeros(len(mesh)).tolist()
    ele_loc = [0.5 * (node[i] + node[i + 1]) for i in range(node_num - 1)]
    ele_loc = np.array(ele_loc)
    wave_num = 16
    k_list = 0.5 * np.sin(2 * np.pi * ele_loc * wave_num) + 0.8
    T = fem_solver.solver(node, mesh, U_BC, U_nodeid, f_BC, f_nodeid, k_list, c_list)
    return node, T


def get_results(don_segment_list, figname, U1=0.15, U2=0.15):
    x_fem, U_fem = single_fem_run(U1, U2)
    # Creat two subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 6))
    axs = axs.flatten()
    # Plot the first subplot
    axs[0].plot(x_fem, U_fem, "black")
    # axs[0].grid()

    for i, don_segment in enumerate(don_segment_list):
        net_name = r"..\\..\\data_driven_training/oned_ell/ode_ex2testv2_sample10000_100meshv2.pth"
        print(don_segment)
        x_don, U_don, noe_x_fem, noe_U_fem, noe_x_don, noe_U_don = single_noe_run(don_segment, U1, U2, net_name)
        relative_L2_error = np.linalg.norm(U_don - U_fem) / np.linalg.norm(U_fem)
        for j in range(len(noe_x_fem)):
            axs[i + 1].plot(noe_x_fem[j], noe_U_fem[j], "blue")
        for j in range(len(noe_x_don)):
            axs[i + 1].plot(noe_x_don[j], noe_U_don[j], "red")
        # Plot the relative L2 error with two effective digits
        axs[i + 1].text(2.5, 0.153, f"Relative L2 error: {relative_L2_error:.2e}", fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.7))
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()

if __name__ == '__main__':
    pass