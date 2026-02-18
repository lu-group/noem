from matplotlib import pyplot as plt
import convexity_test.solver.element.don_element as don_element
import numpy as np
import torch

def get_hessian_positivity(net, u_list, TOL=1e-1):
    device = don_element.DON_info.device
    positivity_list = []
    eig_val_list = []
    energy_list = []
    for u in u_list:
        K = don_element.get_k(u)
        Kg = [[1e7, -1, 0, 0],[-1, 1, 0, 0],[0,0,1,-1],[0,0,-1,1]]
        K = np.array(K)
        Kg = np.array(Kg, dtype=np.float32)
        Kg[1,1] += K[0,0]
        Kg[1,2] += K[0,1]
        Kg[2,1] += K[1,0]
        Kg[2,2] += K[1,1]
        K = Kg[1:,1:]
        print("=" * 50)
        print(f"Hessian at u={u}:\n", K)
        eigenvalues = np.linalg.eigvalsh(K)
        print("Minimum eigenvalue:", np.min(eigenvalues))
        if np.min(eigenvalues) > -1 * TOL:
            positivity_list.append(1)
            print("Positive definite Hessian.")
        else:
            positivity_list.append(0)
            print("Not positive definite Hessian.")
        eig_val_list.append(eigenvalues)
        u_tensor = torch.tensor(u, dtype=torch.float32, requires_grad=True, device=device)
        energy = don_element.one_energy(u_tensor).item()
        energy_list.append(energy)
    return positivity_list, eig_val_list, energy_list


def visualize_eigenvalues(u_list, eig_val_list, num_samples_per_dim):
    u_array = np.array(u_list)
    eig_val_array = np.array(eig_val_list)

    vmin, vmax = -0.3, 0.3

    plt.figure(figsize=(8, 6))

    levels = np.linspace(vmin, vmax, 100)
    cf = plt.contourf(
        u_array[:, 0].reshape(num_samples_per_dim, num_samples_per_dim),
        u_array[:, 1].reshape(num_samples_per_dim, num_samples_per_dim),
        eig_val_array[:, 0].reshape(num_samples_per_dim, num_samples_per_dim),
        levels=levels,
        vmin=vmin,
        vmax=vmax,
        cmap='rainbow'
    )

    cbar = plt.colorbar(cf)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))  # 比如 5 个刻度

    plt.yticks(np.linspace(u_array[:, 1].min(), u_array[:, 1].max(), 5))
    plt.xticks(np.linspace(u_array[:, 0].min(), u_array[:, 0].max(), 5))

    plt.xlabel('U1')
    plt.ylabel('U2')

    plt.plot(
        [-0.5, 0.5, 0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5, 0.5, -0.5],
        color='black',
        linestyle='--',
        linewidth=2,
        alpha=0.7
    )

    plt.rcParams.update({'font.size': 14})

    plt.show()


def visualize_convergency(u_list, convergence_result):
    u_array = np.array(u_list)
    convergence_array = np.array(convergence_result)
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(['red', 'blue'])
    bounds = [-0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(8, 6))

    sc = plt.scatter(
        u_array[:, 0], u_array[:, 1],
        c=convergence_array,
        cmap=cmap,
        norm=norm,
        alpha=0.7
    )
    plt.yticks(np.linspace(u_array[:, 1].min(), u_array[:, 1].max(), 5))
    plt.xticks(np.linspace(u_array[:, 1].min(), u_array[:, 1].max(), 5))
    cbar = plt.colorbar(sc, boundaries=bounds, ticks=[0, 1])
    # cbar.set_label('Convergence (0: No, 1: Yes)')
    # Set all the font size as 14
    plt.rcParams.update({'font.size': 14})
    plt.xlabel('U1')
    plt.ylabel('U2')
    plt.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], color='black', linestyle='--',
             linewidth=2, alpha=0.7)
    plt.show()

if __name__ == "__main__":
    # Set all the font size in the plot as 14
    plt.rcParams.update({'font.size': 14})
    # Uniformly sampled from [-1, 1] x [-1, 1]
    domain_len = 2
    u_min = -domain_len
    u_max = domain_len
    num_samples_per_dim = 21
    u1_values = np.linspace(u_min, u_max, num_samples_per_dim)
    u2_values = np.linspace(u_min, u_max, num_samples_per_dim)
    u_list = []
    for u1 in u1_values:
        for u2 in u2_values:
            u_list.append([u1, u2])
    net_name = r"deeponet.pth"
    don_element.DON_info.initialize(net_name=net_name, device='cpu', x_len=1)
    positivity_list, eig_val_list, energy_list = get_hessian_positivity(don_element.DON_info.net, u_list, TOL=0)
    visualize_eigenvalues(u_list, eig_val_list, num_samples_per_dim)

    from convexity_test.solver.don_solver import convergence_test
    convergence_result = []
    for tu in u_list:
        tu2, tu3 = tu
        tresult = convergence_test(tu2, tu3)
        convergence_result.append(tresult)
    visualize_convergency(u_list, convergence_result)




