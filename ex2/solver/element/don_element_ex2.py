import torch
import numpy as np

class DON_info:
    net = None
    x_len = 1
    y_len = 1
    r = 0.05
    is_zeromean = False
    wave_num = 16
    x0 = 0
    @staticmethod
    def initialize(net_name, x_len, device="cpu"):
        DON_info.net = torch.load(net_name, map_location=device)
        DON_info.device = device
        DON_info.net.eval()
        DON_info.x_len = x_len

def get_k_list(x, wave_num, x0):
    k_list = 0.5 * torch.sin(2 * np.pi * wave_num * (x + x0)) + 0.8
    return k_list
def one_energy(T_BC):
    """
        Compute the energy using Gaussian quadrature and neural network predictions.

        Args:
        T_BC (torch.Tensor): Boundary condition tensor.

        Returns:
        float: Computed energy.
    """
    x_len = DON_info.x_len
    net = DON_info.net
    device = DON_info.device

    num_gauss = 100
    sampling_points = [(i + 0.5) * x_len / num_gauss for i in range(num_gauss)]
    sampling_points = torch.tensor(sampling_points, device=device, dtype=torch.float32, requires_grad=True)
    sampling_points = sampling_points.view(-1, 1)
    k_dis = get_k_list(sampling_points, DON_info.wave_num, DON_info.x0)
    k_dis_tensor = k_dis.view(1, -1)
    weights = torch.ones((sampling_points.shape[0]), 1, device=device) / sampling_points.shape[0]

    T_BC_input = T_BC[1] - T_BC[0]
    T_BC_input = T_BC_input.view(1, -1)

    branch_input1_min = torch.tensor(net.config["branch_input1_min"]).to(device)
    branch_input1_max = torch.tensor(net.config["branch_input1_max"]).to(device)
    branch_input2_min = torch.tensor(net.config["branch_input2_min"]).to(device)
    branch_input2_max = torch.tensor(net.config["branch_input2_max"]).to(device)
    trunk_input_min = torch.tensor(net.config["trunk_input_min"]).to(device)
    trunk_input_max = torch.tensor(net.config["trunk_input_max"]).to(device)
    normalized_branch_input1 = (k_dis_tensor - branch_input1_min) / (branch_input1_max - branch_input1_min)
    normalized_branch_input2 = (T_BC_input - branch_input2_min) / (branch_input2_max - branch_input2_min)
    normalized_trunk_input = (sampling_points - trunk_input_min) / (trunk_input_max - trunk_input_min)
    # input_tensor = torch.cat((normalized_branch_input, normalized_trunk_input), dim=1)
    # net_output = net(input_tensor)
    net_output = net.forward_branch_trunk_fixed([normalized_branch_input1, normalized_branch_input2], normalized_trunk_input)

    output_min = torch.tensor(net.config["output_min"]).to(device)
    output_max = torch.tensor(net.config["output_max"]).to(device)
    T = net_output * (output_max - output_min) + output_min + T_BC[0]

    # Gradient computations
    T_x = torch.autograd.grad(T, sampling_points, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    k_dis = get_k_list(sampling_points, DON_info.wave_num, DON_info.x0)
    f = 0.5
    energy_density = 0.5 * k_dis * T_x ** 2 - f * T
    energy = torch.sum(energy_density * weights) * x_len

    # Start end
    # wave_num = DON_info.wave_num
    # x_1 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True, device=device)
    # normalized_trunk_input_1 = (x_1 - trunk_input_min) / (trunk_input_max - trunk_input_min)
    # net_output_1 = net.forward_branch_trunk_fixed([normalized_branch_input1, normalized_branch_input2], normalized_trunk_input_1)
    # T_1 = net_output_1 * (output_max - output_min) + output_min
    # T_x_1 = torch.autograd.grad(T_1, x_1, grad_outputs=torch.ones_like(T_1), create_graph=True)[0]
    # k_1 = get_k_list(x_1, DON_info.wave_num, DON_info.x0)
    # energy1 = T_x_1 * T_1 * k_1
    # # End end
    # x_2 = torch.tensor([[x_len]], dtype=torch.float32, requires_grad=True, device=device)
    # normalized_trunk_input_2 = (x_2 - trunk_input_min) / (trunk_input_max - trunk_input_min)
    # net_output_2 = net.forward_branch_trunk_fixed([normalized_branch_input1, normalized_branch_input2], normalized_trunk_input_2)
    # T_2 = net_output_2 * (output_max - output_min) + output_min
    # T_x_2 = torch.autograd.grad(T_2, x_2, grad_outputs=torch.ones_like(T_2), create_graph=True)[0]
    # k_2 = get_k_list(x_2, DON_info.wave_num, DON_info.x0)
    # energy2 = -T_x_2 * T_2 * k_2
    # energy += (torch.sum(energy1) + torch.sum(energy2))
    return energy

def get_q(T_BC, wave_num, x0):
    device = DON_info.device
    DON_info.wave_num = wave_num
    DON_info.x0 = x0
    T_BC = np.array(T_BC, dtype=np.float32)
    T_BC = torch.tensor(T_BC, dtype=torch.float32, requires_grad=True, device=device)
    # T_BC: a n x 1 torch.tensor, whose requires_grad is True
    energy = one_energy(T_BC) # A scaler
    # F = dE/d(delta_u)
    q = torch.autograd.grad(energy, T_BC, create_graph=True, retain_graph=True)[0].to("cpu")
    return q.detach().numpy()

def get_k(T_BC, wave_num, x0):
    device = DON_info.device
    DON_info.wave_num = wave_num
    DON_info.x0 = x0
    T_BC = np.array(T_BC, dtype=np.float32)
    T_BC = torch.tensor(T_BC, dtype=torch.float32, requires_grad=True, device=device)
    # Obtain the Hessian matrix of the energy
    K = torch.autograd.functional.hessian(one_energy, T_BC).to("cpu")
    return K.detach().numpy()