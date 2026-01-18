import torch
import numpy as np

class DON_info:
    net = None
    x_len = 1
    r = 0.05
    is_zeromean = False
    f_dist = None
    @staticmethod
    def initialize(net_name, x_len, is_zeromean=True, device="cpu"):
        DON_info.net = torch.load(net_name, map_location=torch.device('cpu'))
        DON_info.net.eval()
        DON_info.device = device
        torch.save(DON_info.net, net_name)
        DON_info.x_len = x_len
        DON_info.is_zeromean = is_zeromean

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
    sampling_points = [(i) / num_gauss for i in range(num_gauss + 1)]
    sampling_points = torch.tensor(sampling_points, device=device, dtype=torch.float32, requires_grad=True) * x_len
    sampling_points = sampling_points.view(-1, 1)

    # Concatenate the filtered points
    weights = 1 / num_gauss

    # T_BC_input = T_BC.repeat(gauss_points_input.shape[0], 1)

    f_dist = DON_info.f_dist
    # f_dist = torch.log(f_dist)
    branch_input1_min = torch.tensor(net.config["branch_input1_min"])
    branch_input1_max = torch.tensor(net.config["branch_input1_max"])
    branch_input2_min = torch.tensor(net.config["branch_input2_min"])
    branch_input2_max = torch.tensor(net.config["branch_input2_max"])
    trunk_input_min = torch.tensor(net.config["trunk_input_min"])
    trunk_input_max = torch.tensor(net.config["trunk_input_max"])
    normalized_branch_input1 = (T_BC - branch_input1_min) / (branch_input1_max - branch_input1_min)
    normalized_branch_input2 = (f_dist - branch_input2_min) / (branch_input2_max - branch_input2_min)
    normalized_trunk_input = (sampling_points - trunk_input_min) / (trunk_input_max - trunk_input_min)

    net_output = net.forward_branch_trunk_fixed(branch_input_list=[normalized_branch_input1, normalized_branch_input2],
                                                trunk_input=normalized_trunk_input)

    output_min = torch.tensor(net.config["output_min"])
    output_max = torch.tensor(net.config["output_max"])
    T = net_output * (output_max - output_min) + output_min

    # Gradient computations
    # T_x = torch.autograd.grad(T, sampling_points, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_x = (T[1:] - T[:-1]) / (x_len / num_gauss)
    # f_dist = torch.exp(f_dist)
    energy_density = 0.5 * f_dist * T_x ** 2 #- f_dist * T
    energy = torch.sum(energy_density * weights) * x_len
    return energy

def get_q(T_BC, f_dist):
    len_T_BC = len(T_BC)
    DON_info.f_dist = torch.tensor(f_dist, dtype=torch.float32).view(1, -1)
    T_BC = np.array(T_BC, dtype=np.float32)
    T_BC = T_BC - np.mean(T_BC)
    if DON_info.is_zeromean:
        T_BC = T_BC - np.mean(T_BC)
    T_BC = torch.tensor(T_BC, dtype=torch.float32).view(1, -1)
    T_BC.requires_grad = True
    # T_BC: a n x 1 torch.tensor, whose requires_grad is True
    energy = one_energy(T_BC) # A scaler
    # F = dE/d(delta_u)
    q = torch.autograd.grad(energy, T_BC, create_graph=True, retain_graph=True)[0]
    q = q.view(len_T_BC)
    return q.detach().numpy(), energy.item()

def get_k(T_BC, f_dist):
    len_T_BC = len(T_BC)
    DON_info.f_dist = torch.tensor(f_dist, dtype=torch.float32).view(1, -1)
    T_BC = np.array(T_BC, dtype=np.float32)
    T_BC = T_BC - np.mean(T_BC)
    if DON_info.is_zeromean:
        T_BC = T_BC - np.mean(T_BC)
    T_BC = torch.tensor(T_BC, dtype=torch.float32).view(1, -1)
    T_BC.requires_grad = True
    # Obtain the Hessian matrix of the energy
    K = torch.autograd.functional.hessian(one_energy, T_BC)
    K = K.view(len_T_BC, len_T_BC)
    return K.detach().numpy(), one_energy(T_BC).item()

if __name__ == "__main__":
    pass
