import torch
import numpy as np

class DON_info:
    net = None
    x_len = 1
    y_len = 1
    r = 0.05
    is_zeromean = False
    @staticmethod
    def initialize(net_name, x_len, k, c, device="cpu"):
        DON_info.net = torch.load(net_name, map_location=device)
        DON_info.device = device
        DON_info.net.eval()
        DON_info.x_len = x_len
        DON_info.k = k
        DON_info.c = c

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
    k = DON_info.k
    c = DON_info.c

    num_gauss = 100
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
    y1_end = T_BC[0]
    y2_end = T_BC[1]
    T = T * sampling_points * (1/8 - sampling_points) + (y2_end - y1_end) / (1 / 8) * sampling_points + y1_end
    # Gradient computations
    T_x = torch.autograd.grad(T, sampling_points, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    x = sampling_points.detach()
    f = 0.5
    wave_num = 16
    k = 0.5 * np.sin(2 * np.pi * x * wave_num) + 0.8
    energy_density = 0.5 * k * T_x ** 2 - f * T
    energy = torch.sum(energy_density * weights) * x_len

    # return energy
    # Left end
    x_1 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True, device=device)
    normalized_trunk_input_1 = (x_1 - trunk_input_min) / (trunk_input_max - trunk_input_min)
    net_output_1 = net.forward_branch_trunk_fixed(normalized_branch_input, normalized_trunk_input_1)
    T_1 = net_output_1 * (output_max - output_min) + output_min
    T_1 = T_1 * x_1 * (1 / 8 - x_1) + (y2_end - y1_end) / (1 / 8) * x_1 + y1_end
    T_x_1 = torch.autograd.grad(T_1, x_1, grad_outputs=torch.ones_like(T_1), create_graph=True)[0]
    k_1 = 0.5 * np.sin(2 * np.pi * 0 * wave_num) + 0.8
    energy1 = -T_x_1 * T_1 * k_1
    # Right end
    x_2 = torch.tensor([[x_len]], dtype=torch.float32, requires_grad=True, device=device)
    normalized_trunk_input_2 = (x_2 - trunk_input_min) / (trunk_input_max - trunk_input_min)
    net_output_2 = net.forward_branch_trunk_fixed(normalized_branch_input, normalized_trunk_input_2)
    T_2 = net_output_2 * (output_max - output_min) + output_min
    T_2 = T_2 * x_2 * (1 / 8 - x_2) + (y2_end - y1_end) / (1 / 8) * x_2 + y1_end
    T_x_2 = torch.autograd.grad(T_2, x_2, grad_outputs=torch.ones_like(T_2), create_graph=True)[0]
    k_2 = 0.5 * np.sin(2 * np.pi * x_len * wave_num) + 0.8
    energy2 = T_x_2 * T_2 * k_2
    energy_total = energy #- 1.0 * (torch.sum(energy1) + torch.sum(energy2))
    return energy_total

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

if __name__ == "__main__":
    def is_positive_definite_eig(matrix, tol=1e-5):
        eigenvalues = np.linalg.eigvalsh(matrix)
        print(eigenvalues)
        return np.all(eigenvalues > -tol)


    net_name = r"..\\..\\..\\data_driven_training/oned_ell/hc_ode_ex2testv2_sample10000_100meshv2.pth"
    DON_info.initialize(net_name=net_name, device='cpu', x_len=1 / 8, k=1, c=1)
    results = []
    for j in range(100):
        u1 = np.random.uniform(-0.05, 0.05)
        u2 = np.random.uniform(-0.05, 0.05)
        T_BC = np.array([u1, u2], dtype=np.float32)
        k = get_k(T_BC)
        tresults = is_positive_definite_eig(k)
        print(k, tresults)
        if not tresults:
            # results.append(tresults)
            pass
        else:
            results.append(tresults)
    print(len(results))
    # for i in range(26):
    #     print(i)
    #     net_name = r"..\\..\\..\\data_driven_training/oned_ell/hc_ode_ex2testv2_verysmall_sam_random_" + str(i) + ".pth"
    #     DON_info.initialize(net_name=net_name, device='cpu', x_len=1 / 8, k=1, c=1)

    # Check if all results are True
    all_positive_definite = all(results)

    print("Number of False results:", 100-len(results))