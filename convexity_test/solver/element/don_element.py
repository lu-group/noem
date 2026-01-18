import torch
import numpy as np

class DON_info:
    net = None
    x_len = 1
    is_zeromean = False
    @staticmethod
    def initialize(net_name, x_len, device="cpu"):
        DON_info.net = torch.load(net_name, map_location=device)
        DON_info.device = device
        DON_info.net.eval()
        DON_info.x_len = x_len

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

    num_gauss = 50
    sampling_points = [(i + 0.5) / num_gauss for i in range(num_gauss)]
    sampling_points = torch.tensor(sampling_points, device=device, dtype=torch.float32, requires_grad=True) * x_len
    sampling_points = sampling_points.view(-1, 1)
    weights = torch.ones((sampling_points.shape[0]), 1, device=device) / sampling_points.shape[0]

    T_BC_input = T_BC.view(1,-1)

    branch_input_min = torch.tensor(net.config["branch_input_min"]).to(device)
    branch_input_max = torch.tensor(net.config["branch_input_max"]).to(device)
    trunk_input_min = torch.tensor(net.config["trunk_input_min"]).to(device)
    trunk_input_max = torch.tensor(net.config["trunk_input_max"]).to(device)
    normalized_branch_input = (T_BC_input - branch_input_min) / (branch_input_max - branch_input_min)
    normalized_trunk_input = (sampling_points - trunk_input_min) / (trunk_input_max - trunk_input_min)
    # input_tensor = torch.cat((normalized_branch_input, normalized_trunk_input), dim=1)
    # net_output = net(input_tensor)
    net_output = net.forward_branch_trunk_fixed(normalized_branch_input, normalized_trunk_input)

    output_min = torch.tensor(net.config["output_min"]).to(device)
    output_max = torch.tensor(net.config["output_max"]).to(device)
    T = net_output * (output_max - output_min) + output_min

    # Gradient computations
    T_x = torch.autograd.grad(T, sampling_points, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    x = sampling_points.detach()
    f = 0
    k = 1 + x * (x - 1)
    c = 0
    energy_density = 0.5 * k * T_x ** 2 + 0.5 * c * T ** 2 - f * T
    energy = torch.sum(energy_density * weights) * x_len
    return energy

def get_q(T_BC):
    device = DON_info.device
    T_BC = np.array(T_BC, dtype=np.float32)
    T_BC = torch.tensor(T_BC, dtype=torch.float32, requires_grad=True, device=device)
    # T_BC: a n x 1 torch.tensor, whose requires_grad is True
    energy = one_energy(T_BC) # A scaler
    # F = dE/d(delta_u)
    q = torch.autograd.grad(energy, T_BC, create_graph=True, retain_graph=True)[0].to("cpu")
    return q.detach().numpy()

def get_k(T_BC):
    device = DON_info.device
    T_BC = np.array(T_BC, dtype=np.float32)
    T_BC = torch.tensor(T_BC, dtype=torch.float32, requires_grad=True, device=device)
    # Obtain the Hessian matrix of the energy
    K = torch.autograd.functional.hessian(one_energy, T_BC).to("cpu")
    return K.detach().numpy()