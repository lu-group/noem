import torch
import numpy as np

class DON_info:
    net = None
    x_len = 1
    y_len = 1
    r = 0.05
    is_zeromean = False
    K_dist = None
    @staticmethod
    def initialize(net_name, x_len, y_len, r=0, is_zeromean=False):
        DON_info.net = torch.load(net_name, map_location=torch.device('cpu'))
        DON_info.net.eval()
        torch.save(DON_info.net, net_name)
        DON_info.x_len = x_len
        DON_info.y_len = y_len
        DON_info.r = r
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
    y_len = DON_info.y_len
    r = DON_info.r
    net = DON_info.net

    num_gauss = 20
    gauss_points = [(i + 0.5) / num_gauss for i in range(num_gauss)]
    gauss_points = torch.tensor(gauss_points)

    # Create 2D grid of points and weights
    # gx, gy = torch.meshgrid(gauss_points, gauss_points, indexing='ij')
    gy, gx = torch.meshgrid(gauss_points, gauss_points, indexing='ij')
    # Generate and preprocess Gaussian points
    gauss_points_x_input = (gx * x_len).flatten().unsqueeze(-1).requires_grad_(True)
    gauss_points_y_input = (gy * y_len).flatten().unsqueeze(-1).requires_grad_(True)

    # Concatenate the filtered points
    gauss_points_input = torch.cat([gauss_points_x_input, gauss_points_y_input], dim=1)

    gauss_weights_2D = torch.ones((gauss_points_input.shape[0]), 1) / gauss_points_input.shape[0]

    # T_BC_input = T_BC.repeat(gauss_points_input.shape[0], 1)

    K_dist = DON_info.K_dist

    branch_input1_min = torch.tensor(net.config["branch_input1_min"])
    branch_input1_max = torch.tensor(net.config["branch_input1_max"])
    branch_input2_min = torch.tensor(net.config["branch_input2_min"])
    branch_input2_max = torch.tensor(net.config["branch_input2_max"])
    trunk_input_min = torch.tensor(net.config["trunk_input_min"])
    trunk_input_max = torch.tensor(net.config["trunk_input_max"])
    normalized_branch_input1 = (K_dist - branch_input1_min) / (branch_input1_max - branch_input1_min)
    normalized_branch_input2 = (T_BC - branch_input2_min) / (branch_input2_max - branch_input2_min)
    normalized_trunk_input = (gauss_points_input - trunk_input_min) / (trunk_input_max - trunk_input_min)

    # input_tensor = torch.cat((normalized_branch_input, normalized_trunk_input), dim=1)
    # print(torch.sum(gauss_weights_2D))

    # Neural network prediction
    net_output = net.forward_branch_trunk_fixed(branch_input_list=[normalized_branch_input1, normalized_branch_input2],
                                                trunk_input=normalized_trunk_input)

    output_min = torch.tensor(net.config["output_min"])
    output_max = torch.tensor(net.config["output_max"])
    T = net_output * (output_max - output_min) + output_min

    # Gradient computations
    T_x = torch.autograd.grad(T, gauss_points_x_input, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, gauss_points_y_input, grad_outputs=torch.ones_like(T), create_graph=True)[0]

    energy_density = 0.5 * (T_x ** 2 + T_y ** 2) * torch.tensor(DON_info.K_dist).view(-1, 1)
    area = x_len * y_len
    energy = torch.sum(energy_density * gauss_weights_2D) * area
    return energy

def get_q(T_BC, K_dist):
    len_T_BC = len(T_BC)
    DON_info.K_dist = torch.tensor(K_dist, dtype=torch.float32).view(1, -1)
    T_BC = np.array(T_BC, dtype=np.float32)
    T_BC = T_BC - np.mean(T_BC)
    # if DON_info.is_zeromean:
    #     T_BC = T_BC - np.mean(T_BC)
    T_BC = torch.tensor(T_BC, dtype=torch.float32).view(1, -1)
    T_BC.requires_grad = True
    # T_BC: a n x 1 torch.tensor, whose requires_grad is True
    energy = one_energy(T_BC) # A scaler
    # F = dE/d(delta_u)
    q = torch.autograd.grad(energy, T_BC, create_graph=True, retain_graph=True)[0]
    q = q.view(len_T_BC)
    return q.detach().numpy()

def get_k(T_BC, K_dist):
    len_T_BC = len(T_BC)
    DON_info.K_dist = torch.tensor(K_dist, dtype=torch.float32).view(1, -1)
    T_BC = np.array(T_BC, dtype=np.float32)
    T_BC = T_BC - np.mean(T_BC)
    # if DON_info.is_zeromean:
    #     T_BC = T_BC - np.mean(T_BC)
    T_BC = torch.tensor(T_BC, dtype=torch.float32).view(1, -1)
    T_BC.requires_grad = True
    # Obtain the Hessian matrix of the energy
    K = torch.autograd.functional.hessian(one_energy, T_BC)
    K = K.view(len_T_BC, len_T_BC)
    return K.detach().numpy()

if __name__ == "__main__":
    pass
