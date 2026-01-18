# operator net element solver
# PDE: d2u/dx2 = 0; BC: u(0) = 0, du/dx(3) = P/EA
# exact solution: u = P/EA * x

# External Libs.
import torch
import torch.nn as nn
import numpy as np

import util.integral as integral
import util.gradients as gradients

def fem_eleK(x1, x2, EA=1):
    """Linear 1D bar element stiffness matrix."""
    length = abs(x2 - x1)
    return np.array([[1, -1], [-1, 1]]) * EA / length

def one_energy(delta_u, EA_list, net, EA=1, n=50):
    """Element energy predicted by DeepONet for the middle element."""
    x = torch.linspace(0, 1, n, requires_grad=True)
    delta_u_input = torch.ones(n, 1) * delta_u
    EA_input = torch.Tensor.repeat(EA_list, n, 1)
    model_input = torch.cat(
        (EA_input.view(n, 50), delta_u_input.view(n, 1), x.view(n, 1)), dim=1
    )
    input_scaler = net.config["input_scaler"]
    input_scaler = torch.tensor([input_scaler], dtype=torch.float32)
    model_input = model_input / input_scaler
    output = net(model_input)
    output_scaler = net.config["output_scaler"]
    output_scaler = torch.tensor([output_scaler], dtype=torch.float32)
    u = output * output_scaler
    u_x = gradients.gradients(u, x)
    energy = 0.5 * torch.sum(EA_list * u_x ** 2) / n
    return energy

def oneF(delta_u, EA_list, net, EA=1):
    """Element force derivative wrt delta_u."""
    # delta_u = torch.tensor(delta_u, dtype=torch.float32, requires_grad=True)
    # EA_list = torch.tensor(EA_list, dtype=torch.float32)
    energy = one_energy(delta_u, EA_list, net, EA)
    # F = dE/d(delta_u)
    F = torch.autograd.grad(energy, delta_u, create_graph=True, retain_graph=True)[0]
    return F

def oneF2(delta_u, EA_list, net, EA=1):
    """Element nodal force vector from DeepONet energy."""
    delta_u = torch.tensor(delta_u, dtype=torch.float32, requires_grad=True)
    EA_list = torch.tensor(EA_list, dtype=torch.float32)
    energy = one_energy(delta_u, EA_list, net, EA)
    # F = dE/d(delta_u)
    F = torch.autograd.grad(energy, delta_u, create_graph=True, retain_graph=True)[0]
    FVALUE = F.item()
    F = np.array([-FVALUE, FVALUE])
    return F

def oneK(u1, u2, EA_list, net):
    """Element stiffness derived from DeepONet energy."""
    delta_u = u2 - u1
    delta_u = torch.tensor(delta_u, dtype=torch.float32, requires_grad=True)
    EA_list = torch.tensor(EA_list, dtype=torch.float32)
    F = oneF(delta_u, EA_list, net)
    K = torch.autograd.grad(F, delta_u, create_graph=True, retain_graph=True)[0]
    K = K.item()
    return np.array([[K, -K], [-K, K]])

def solver(P, EA_list, net):
    appF = np.array([0, 0, 0, P])
    unbF = np.array([0, 0, 0, P])
    U = np.zeros(4)
    iter = 0
    while True:
        K = np.zeros((4, 4))
        ele_k = []
        for i in range(3):
            if i != 1:
                ele_k.append(fem_eleK(i, i + 1))
            else:
                ele_k.append(oneK(U[1], U[2], EA_list, net))
        for i in range(3):
            K[i:i + 2, i:i + 2] += ele_k[i]
        K[0, :] = 0
        K[:, 0] = 0
        K[0, 0] = 1
        unbF[0] = 0
        deltaU = np.linalg.solve(K, unbF)

        U += deltaU
        resF = np.zeros(4)
        # First element
        ele_k = fem_eleK(0, 1)
        tU = U[0:2]
        ele_F = np.dot(ele_k, tU)
        resF[0:2] += ele_F
        # Second element
        ele_F = oneF2(U[2] - U[1], EA_list, net)
        resF[1:3] += ele_F
        # Third element
        ele_k = fem_eleK(2, 3)
        tU = U[2:4]
        ele_F = np.dot(ele_k, tU)
        resF[2:4] += ele_F

        unbF = appF - resF
        if np.linalg.norm(deltaU) / np.linalg.norm(U) < 1e-3:
            break
        else:
            print("TOL:", np.linalg.norm(deltaU) / np.linalg.norm(U))
        iter += 1
        if iter > 5:
            break
    return U

def visual(U, P, EA_list, net, n=100, is_show=True):
    x = np.linspace(0, 3, n)
    u_pre = []
    for i in range(n):
        if x[i] < 1:
            u_pre.append(U[0] + (U[1] - U[0]) * x[i])
        elif 1 <= x[i] and x[i] < 2:
            delta_u = U[2] - U[1]
            tx = x[i] - 1
            tEA_list = list(EA_list)
            model_input = tEA_list + [delta_u, tx]
            model_input = torch.tensor(model_input, dtype=torch.float32)
            input_scaler = net.config["input_scaler"]
            input_scaler = torch.tensor([input_scaler], dtype=torch.float32)
            model_input = model_input / input_scaler
            output = net(model_input)
            output_scaler = net.config["output_scaler"]
            output_scaler = torch.tensor([output_scaler], dtype=torch.float32)
            u = output * output_scaler
            u_pre.append(U[1] + u.item())
        else:
            u_pre.append(U[2] + (U[3] - U[2]) * (x[i] - 2))
    import matplotlib.pyplot as plt
    # Plot two subplots in one column
    plt.figure()
    plt.subplot(211)
    # Plot EA_list
    EA_list = EA_list.tolist()
    EA_list = [1,1] + EA_list + [1,1]
    x_list = np.linspace(1, 2, 50).tolist()
    x_list = [0,0.999] + x_list + [2.001,3]
    # Plot EA in the first subplot
    plt.plot(x_list, EA_list)
    plt.legend()
    plt.ylabel("k")
    plt.xlabel("x")
    # CHange the size of the first subplot
    plt.subplot(212)
    plt.plot(x, u_pre, label="D-NOE Results")
    x_acc, u_acc = exact(P, EA_list, n)
    plt.plot(x_acc, u_acc, label="FEM Results", linestyle="--")
    plt.legend()
    plt.ylabel("y")
    plt.xlabel("x")
    relative_L2_error = (
        np.linalg.norm(np.array(u_pre) - np.array(u_acc))
        / np.linalg.norm(np.array(u_acc))
    )
    print(relative_L2_error)
    # Plot the relative L2 error with two effective digits
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(u_pre), np.max(u_pre)
    x_loc = x_min + 0.8 * (x_max - x_min)
    y_loc = y_min + 0.03 * (y_max - y_min)
    plt.text(x_loc, y_loc, f'Relative L2 error: {relative_L2_error:.2e}', horizontalalignment='center', bbox=bbox_props, )
    if is_show:
        plt.show()
    plt.close()
    return relative_L2_error

def exact(P, EA_list, num):
    """Finite element benchmark with fixed mesh size."""
    length = 3.0
    x = np.linspace(0.0, length, num)
    n_nodes = len(x)
    K = np.zeros((n_nodes, n_nodes))
    f = np.zeros(n_nodes)
    # Neumann BC at x=3: EA * u' = P -> nodal force P at last node.
    f[-1] = P

    for i in range(n_nodes - 1):
        x1, x2 = x[i], x[i + 1]
        xm = 0.5 * (x1 + x2)
        if xm < 1 or xm >= 2:
            EA = 1.0
        else:
            indices = (xm - 1.0) * 50
            ti = int(indices)
            tj = min(ti + 1, len(EA_list) - 1)
            EA = EA_list[ti] + (EA_list[tj] - EA_list[ti]) * (indices - ti)
        k_e = fem_eleK(x1, x2, EA)
        K[i:i + 2, i:i + 2] += k_e

    # Apply Dirichlet BC: u(0) = 0
    K[0, :] = 0.0
    K[:, 0] = 0.0
    K[0, 0] = 1.0
    f[0] = 0.0

    u = np.linalg.solve(K, f)
    return list(x), list(u)
from scipy.stats import multivariate_normal

def grf_1Dv2(x, l):
    # Covariance matrix using the Gaussian kernel
    def gaussian_kernel(x, l):
        """ Generate a Gaussian kernel matrix with correlation length l. """
        x = x[:, np.newaxis]  # Convert to column vector
        # Squared exponential kernel
        return np.exp(-0.5 * (x - x.T) ** 2 / l ** 2)

    covariance_matrix = gaussian_kernel(x, l)

    # Sample from the multivariate normal distribution
    mean = np.zeros(len(x))  # Zero mean
    random_field = multivariate_normal.rvs(mean=mean, cov=covariance_matrix)
    return x, random_field

if __name__ == '__main__':
    # Fix random seed
    np.random.seed(12345)
    import grf as grf
    ele_loc = [0.5 * (1 / 50) for i in range(50)]
    ele_loc = np.array(ele_loc)
    _, EA_list = grf.grf(0, 1, 50, 0.3)
    EA_list = np.exp(EA_list)
    deeponet = torch.load(r"ex1.2_don.pt", map_location="cpu")
    P = 0.1
    u = solver(P, EA_list, deeponet)
    error = visual(u, P, EA_list, deeponet)


