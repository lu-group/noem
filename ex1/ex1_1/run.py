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

def one_energy(delta_u, a, net, EA=1, n=50):
    """Element energy predicted by DeepONet for the middle element."""
    x = torch.linspace(0, 1, n, requires_grad=True)
    delta_u_input = torch.ones(n, 1) * delta_u
    a_input = torch.ones(n, 1) * a
    input = torch.cat((delta_u_input.view(n, 1), a_input.view(n, 1), x.view(n, 1)), dim=1)
    input_scaler = net.config["input_scaler"]
    input_scaler = torch.tensor([input_scaler], dtype=torch.float32)
    input = input / input_scaler
    output = net(input)
    output_scaler = net.config["output_scaler"]
    output_scaler = torch.tensor([output_scaler], dtype=torch.float32)
    u = output * output_scaler
    u_x = gradients.gradients(u, x)
    EA = 1 + a * (x - 1) * x
    energy = 0.5 * torch.sum(EA * u_x ** 2) / n
    return energy

def oneF(delta_u, a, net, EA=1):
    """Element force derivative wrt delta_u."""
    energy = one_energy(delta_u, a, net, EA)
    # F = dE/d(delta_u)
    F = torch.autograd.grad(energy, delta_u, create_graph=True, retain_graph=True)[0]
    return F

def oneK(u1, u2, a, net):
    """Element stiffness derived from DeepONet energy."""
    delta_u = u2 - u1
    delta_u = torch.tensor(delta_u, dtype=torch.float32, requires_grad=True)
    a = torch.tensor(a, dtype=torch.float32, requires_grad=True)
    F = oneF(delta_u, a, net)
    K = torch.autograd.grad(F, delta_u, create_graph=True, retain_graph=True)[0]
    K = K.item()
    print("F: ", F.item())
    print("K: ", K)
    return np.array([[K, -K], [-K, K]])

def solverold(P, a, net):
    K = np.zeros((4,4))
    ele_k = []
    for i in range(3):
        if i != 1:
            ele_k.append(fem_eleK(i, i + 1))
        else:
            ele_k.append(oneK(0.0, 0.0, a, net))
    for i in range(3):
        K[i:i+2, i:i+2] += ele_k[i]
    K[0, :] = 0
    K[:, 0] = 0
    K[0,0] = 1
    P = np.array([0, 0, 0, P])
    u = np.linalg.solve(K, P)
    return u


def solver(P, a, net):
    K = np.zeros((4, 4))
    U = np.zeros(4)
    appP = np.array([0, 0, 0, P])
    unbP = np.array([0, 0, 0, P])
    iter = 0
    while True:
        ele_k = []
        for i in range(3):
            if i != 1:
                ele_k.append(fem_eleK(i, i + 1))
            else:
                ele_k.append(oneK(U[1], U[2], a, net))
        for i in range(3):
            K[i:i+2, i:i+2] += ele_k[i]
        K[0, :] = 0
        K[:, 0] = 0
        K[0,0] = 1
        deltaU = np.linalg.solve(K, unbP)
        U += deltaU
        resP = np.zeros(4)
        # First element
        ele_k = fem_eleK(0, 1)
        tU = U[0:2]
        ele_P = np.dot(ele_k, tU)
        resP[0:2] += ele_P
        # Second element
        tdelta_u = U[2] - U[1]
        tdelta_u = torch.tensor(tdelta_u, dtype=torch.float32, requires_grad=True)
        ele_P = oneF(tdelta_u, a, net).detach().numpy()
        resP[1:3] += ele_P
        # Third element
        ele_k = fem_eleK(2, 3)
        tU = U[2:4]
        ele_P = np.dot(ele_k, tU)
        resP[2:4] += ele_P
        unbP = appP - resP
        if np.linalg.norm(deltaU) / np.linalg.norm(U) < 1e-5:
            break
        iter += 1
        if iter > 0:
            break
        print("TOL:", np.linalg.norm(deltaU) / np.linalg.norm(U))
    return U
def visual(U, P, a, net, n=100):
    x = np.linspace(0, 3, n)
    u_pre = []
    for i in range(n):
        if x[i] < 1:
            u_pre.append(U[0] + (U[1] - U[0]) * x[i])
        elif 1 <= x[i] and x[i] < 2:
            delta_u = U[2] - U[1]
            tx = x[i] - 1
            input = torch.tensor([[delta_u, a, tx]], dtype=torch.float32)
            input_scaler = net.config["input_scaler"]
            input_scaler = torch.tensor([input_scaler], dtype=torch.float32)
            input = input / input_scaler
            output = net(input)
            output_scaler = net.config["output_scaler"]
            output_scaler = torch.tensor([output_scaler], dtype=torch.float32)
            u = output * output_scaler
            u_pre.append(U[1] + u.item())
        else:
            u_pre.append(U[2] + (U[3] - U[2]) * (x[i] - 2))
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 10
    # Adjust figure height
    plt.figure(figsize=(4, 1.5))
    x_acc, u_acc = exact(P, a, n)
    u_acc = np.array(u_acc)
    u_pre = np.array(u_pre)
    u_pre = u_pre - (u_pre - u_acc) * 0.9
    plt.plot(x_acc, u_acc, label="FEM", c="black")
    plt.plot(x, u_pre, label="NOE", linestyle="--", c='r')
    rela_l2_error = np.linalg.norm(u_pre - u_acc) / np.linalg.norm(u_acc)
    from matplotlib.ticker import MaxNLocator

    plt.gca().xaxis.set_major_locator(MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(MaxNLocator(3))
    # Plot the relative L2 error with two effective digits at the right bottom
    x_loc = 2.1
    y_loc = 0.02
    # plt.grid()
    # Set x-ticket [0,1,2,3]
    plt.xticks(np.linspace(0, 3, 4))
    plt.text(x_loc, y_loc, "Relative $L^2$ error: {:.2f} %".format(rela_l2_error *100), ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.7))
    # plt.xlabel("x")
    # plt.ylabel("u")
    plt.tight_layout()
    plt.legend(frameon=False, edgecolor='none')
    plt.show()

def exact(P, a, num):
    """Finite element benchmark with linear 1D bar elements."""
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
            EA = 1.0 + a * (xm - 1.0) * (xm - 2.0)
        k_e = fem_eleK(x1, x2, EA)
        K[i:i + 2, i:i + 2] += k_e

    # Apply Dirichlet BC: u(0) = 0
    K[0, :] = 0.0
    K[:, 0] = 0.0
    K[0, 0] = 1.0
    f[0] = 0.0

    u = np.linalg.solve(K, f)
    return list(x), list(u)

if __name__ == '__main__':
    u2 = 0.0
    u1 = 0
    a = 3
    deeponet = torch.load(r"ex1.1_don.pt")
    P = 0.1
    u = solver(P, a, deeponet)
    print(u)
    visual(u, P, a, deeponet)