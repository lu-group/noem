import torch
import numpy as np

class DON_info:
    net = None
    x_len = 1
    y_len = 1
    r = 0.05
    is_zeromean = False
    @staticmethod
    def initialize(net_name, x_len, y_len, r=0.05, is_zeromean=False, device="cpu"):
        DON_info.net = torch.load(net_name, map_location=device)
        DON_info.device = device
        DON_info.net.eval()
        torch.save(DON_info.net, net_name)
        DON_info.x_len = x_len
        DON_info.y_len = y_len
        DON_info.r = r
        DON_info.is_zeromean = is_zeromean

def forward_branch_trunk_fixed(net, branch_input, trunk_input):
    branch_output = net.branch_net(branch_input)
    trunk_output = net.trunk_net(trunk_input)

    expanded_branch_output = torch.cat(
        [branch_output[i].unsqueeze(0).repeat(len(trunk_input), 1) for i in range(len(branch_input))],
        dim=0)
    # Repeat n trunk_output for each branch_output
    expanded_trunk_output = trunk_output.repeat(len(branch_input), 1)

    out = batch_segmented_dot_product(expanded_branch_output, expanded_trunk_output, net.model_channel_size)
    return out

def batch_segmented_dot_product(branch_output, trunk_output, segment_sizes):
    output_list = [branch_output] + [trunk_output]
    stacked_tensors = torch.stack(output_list)
    mul_tensors = torch.prod(stacked_tensors, dim=0)
    # Sum the mul_tensors at each row according to the segment_sizes
    results = torch.zeros(len(trunk_output), len(segment_sizes), device=trunk_output.device)
    start_idx = 0
    for j in range(len(segment_sizes)):
        length = segment_sizes[j]
        results[:, j] = mul_tensors[:, start_idx:start_idx + length].sum(dim=1)
        start_idx += length
    return results

def one_energy(T_BC):
    """
        Compute the energy using Gaussian quadrature and neural network predictions.

        Args:
        T_BC (torch.Tensor): Boundary condition tensor.

        Returns:
        float: Computed energy.
    """
    x_len = DON_info.x_len
    y_len = DON_info.y_len
    r = DON_info.r
    net = DON_info.net
    device = DON_info.device

    num_gauss = 50
    gauss_points = [(i + 0.5) / num_gauss for i in range(num_gauss)]
    gauss_points = torch.tensor(gauss_points, device=device, dtype=torch.float32)

    # Create 2D grid of points and weights
    gx, gy = torch.meshgrid(gauss_points, gauss_points, indexing='ij')

    # Generate and preprocess Gaussian points
    gauss_points_x_input = (gx * x_len).flatten().unsqueeze(-1)
    gauss_points_y_input = (gy * y_len).flatten().unsqueeze(-1)
    # Create mask
    mask = (gauss_points_x_input - 0.5 * x_len) ** 2 + (gauss_points_y_input - 0.5 * y_len) ** 2 < r ** 2
    mask = mask.flatten()
    # Filter out points within the circle
    gauss_points_x_input_filtered = gauss_points_x_input[~mask].detach().requires_grad_(True)
    gauss_points_y_input_filtered = gauss_points_y_input[~mask].detach().requires_grad_(True)

    # Concatenate the filtered points
    gauss_points_input = torch.cat([gauss_points_x_input_filtered, gauss_points_y_input_filtered], dim=1)

    gauss_weights_2D = torch.ones((gauss_points_input.shape[0]), 1, device=device) / gauss_points_input.shape[0]

    T_BC_input = T_BC.view(1,-1)

    branch_input_min = torch.tensor(net.config["branch_input_min"]).to(device)
    branch_input_max = torch.tensor(net.config["branch_input_max"]).to(device)
    trunk_input_min = torch.tensor(net.config["trunk_input_min"]).to(device)
    trunk_input_max = torch.tensor(net.config["trunk_input_max"]).to(device)
    normalized_branch_input = (T_BC_input - branch_input_min) / (branch_input_max - branch_input_min)
    normalized_trunk_input = (gauss_points_input - trunk_input_min) / (trunk_input_max - trunk_input_min)
    # input_tensor = torch.cat((normalized_branch_input, normalized_trunk_input), dim=1)
    # net_output = net(input_tensor)
    net_output = forward_branch_trunk_fixed(net, normalized_branch_input, normalized_trunk_input)

    output_min = torch.tensor(net.config["output_min"]).to(device)
    output_max = torch.tensor(net.config["output_max"]).to(device)
    T = net_output * (output_max - output_min) + output_min

    # Gradient computations
    T_x = torch.autograd.grad(T, gauss_points_x_input_filtered, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, gauss_points_y_input_filtered, grad_outputs=torch.ones_like(T), create_graph=True)[0]

    energy_density = (T_x ** 2 + T_y ** 2) * 0.5
    area = x_len * y_len - np.pi * r ** 2
    energy = torch.sum(energy_density * gauss_weights_2D) * area
    return energy

def get_q(T_BC):
    device = DON_info.device
    T_BC = np.array(T_BC, dtype=np.float32)
    if DON_info.is_zeromean:
        T_BC = T_BC - np.mean(T_BC)
    T_BC = torch.tensor(T_BC, dtype=torch.float32, requires_grad=True, device=device)
    # T_BC: a n x 1 torch.tensor, whose requires_grad is True
    energy = one_energy(T_BC) # A scaler
    # F = dE/d(delta_u)
    q = torch.autograd.grad(energy, T_BC, create_graph=True, retain_graph=True)[0].to("cpu")
    return q.detach().numpy()

def get_k(T_BC):
    device = DON_info.device
    T_BC = np.array(T_BC, dtype=np.float32)
    if DON_info.is_zeromean:
        T_BC = T_BC - np.mean(T_BC)
    T_BC = torch.tensor(T_BC, dtype=torch.float32, requires_grad=True, device=device)
    # Obtain the Hessian matrix of the energy
    K = torch.autograd.functional.hessian(one_energy, T_BC).to("cpu")
    return K.detach().numpy()