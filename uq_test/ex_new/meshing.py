import numpy as np

def get_f(x):
    x = x % 1
    return x * (1 - x)

def get_fem_mesh(L, ele_num, k=1, c=1):
    node = np.linespace(0, L, ele_num + 1)
    element = [[i, i + 1] for i in range(ele_num)]
    k_list = np.ones(len(element)).tolist()
    c_list = np.ones(len(element)).tolist()
    f_list = get_f(node) * L / ele_num
    return node, element, k_list, c_list, f_list

def get_don_mesh(L, ele_num, don_segment, k=1, c=1):
    node = np.linspace(0, L, ele_num + 1)
    for i in range(len(don_segment)):
        tx1, tx2 = don_segment[i]
        deleted_idx = []
        for j in range(len(node)):
            x = node[j]
            if tx1 < x < tx2:
                deleted_idx.append(j)
        node = np.delete(node, deleted_idx)
    deleted_idx = []
    fem_element = [[i, i + 1] for i in range(len(node) - 1)]
    for i in range(len(fem_element)):
        x1 = node[fem_element[i][0]]
        x2 = node[fem_element[i][1]]
        x = 0.5 * (x1 + x2)
        for j in range(len(don_segment)):
            tx1, tx2 = don_segment[j]
            if tx1 < x < tx2:
                deleted_idx.append(i)
    fem_element = np.array(fem_element)
    don_element = fem_element[deleted_idx]
    fem_element = np.delete(fem_element, deleted_idx, axis=0)
    k_list = np.ones(len(fem_element)).tolist()
    c_list = np.ones(len(fem_element)).tolist()
    f_list = get_f(node) * L / ele_num
    return node, fem_element, don_element, k_list, c_list, f_list
import torch
def get_don_prediction(U1, U2, x, net):
    x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    x = x % 1
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

def is_don(x, segment):
    for i in range(len(segment)):
        if segment[i][0] < x < segment[i][1]:
            return i
    return -1
if __name__ == '__main__':
    L = 10
    ele_num = 500
    don_segment = [[1, 2], [4, 5], [7, 8],[9,10]]

    node, fem_mesh, don_mesh, k_list, c_list, f_list = get_don_mesh(L, ele_num, don_segment)
    import ex2.solver.don_solver as don_solver
    import ex2.solver.element.don_element as don_element
    net_name = r"..\\..\\data_driven_training/oned_ell/ode_sample10000_100meshv2.pth"
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1, k=1, c=1)
    U = don_solver.one_solver(node=node, mesh=fem_mesh, U_BC=[0,0], U_nodeid=[0, len(node)-1],
                              f_BC=f_list, f_nodeid=range(len(node)), U_oneBC_nodeid_list=don_mesh,
                              k_list=k_list, c_list=k_list, don_element=don_element, is_logging=True)
    x = np.linspace(0, L, 501)
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

    import matplotlib.pyplot as plt
    plt.plot(x, U_pred, 'r')
    plt.grid(True)
    plt.show()



