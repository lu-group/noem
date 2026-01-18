import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from matplotlib.ticker import MaxNLocator

colormap = "rainbow"
fontsize = 12
plt.rcParams['font.size'] = fontsize
figsize = (5,4)

def visual_mesh(node, element):
    plt.figure()
    print("Plotting mesh...")
    for tri in element:
        triangle = plt.Polygon(node[tri, :2], edgecolor='black', facecolor='none')
        plt.gca().add_patch(triangle)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([np.min(node[:, 0]), np.max(node[:, 0])])
    plt.ylim([np.min(node[:, 1]), np.max(node[:, 1])])
    plt.grid(True)
    plt.show()

def plot_2d_contour(node, results, x0_list, y0_list, r_list, vmin=None, vmax=None, figname="contour.png",
                    node_visual=False, is_show=True, title=""):
    grid_x, grid_y = np.mgrid[
                     np.min(node[:, 0]):np.max(node[:, 0]):300j,  # Increase resolution for smoother edges
                     np.min(node[:, 1]):np.max(node[:, 1]):300j
                     ]

    grid_z = griddata(node, results.ravel(), (grid_x, grid_y), method='linear')

    mask = np.zeros(grid_z.shape, dtype=bool)
    for x0, y0, r in zip(x0_list, y0_list, r_list):
        mask |= (grid_x - x0) ** 2 + (grid_y - y0) ** 2 <= r ** 2
    grid_z[mask] = np.nan  # Mask out the circles

    plt.figure(figsize=figsize)
    levels = 100
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=colormap, vmin=vmin, vmax=vmax)

    if vmin is None or vmax is None:
        vmin, vmax = np.nanmin(grid_z), np.nanmax(grid_z)
    colorbar = plt.colorbar(contour, boundaries=np.linspace(vmin, vmax, levels))
    colorbar.locator = MaxNLocator(nbins=4)
    colorbar.update_ticks()
    plt.gca().xaxis.set_major_locator(MaxNLocator(4))
    plt.gca().yaxis.set_major_locator(MaxNLocator(4))
    for x0, y0, r in zip(x0_list, y0_list, r_list):
        circle = plt.Circle((x0, y0), r, color='white', fill=False, linestyle='--')
        plt.gca().add_patch(circle)

    if node_visual:
        plt.scatter(node[:, 0], node[:, 1], c='red', s=5, label='Nodes')
        for x0, y0 in zip(x0_list, y0_list):
            plt.scatter([x0], [y0], c='blue', s=50, label='Circle Centers')
    plt.title(title)
    plt.tight_layout()
    plt.legend(loc='best') if node_visual else None
    plt.savefig(figname)
    if is_show:
        plt.show()
    plt.close()

def plot_2d_diff_contour(node1, results1, node2, results2, x0_list, y0_list, r_list, title=None,
                         vmin=None, vmax=None, figname="contour_diff.png", is_show=True):
    grid_x, grid_y = np.mgrid[
                     np.min(node1[:, 0]):np.max(node1[:, 0]):300j,  # Increase resolution for smoother edges
                     np.min(node1[:, 1]):np.max(node1[:, 1]):300j
                     ]

    grid_z1 = griddata(node1, results1.ravel(), (grid_x, grid_y), method='linear')
    grid_z2 = griddata(node2, results2.ravel(), (grid_x, grid_y), method='linear')
    grid_z = np.abs(grid_z1 - grid_z2)

    mask = np.zeros(grid_z.shape, dtype=bool)
    for x0, y0, r in zip(x0_list, y0_list, r_list):
        mask |= (grid_x - x0) ** 2 + (grid_y - y0) ** 2 <= r ** 2
    grid_z[mask] = np.nan  # Mask out the circles

    plt.figure(figsize=figsize)
    levels = 100
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=colormap, vmin=vmin, vmax=vmax)

    if vmin is None or vmax is None:
        vmin, vmax = np.nanmin(grid_z), np.nanmax(grid_z)
    colorbar = plt.colorbar(contour, boundaries=np.linspace(vmin, vmax, levels))
    colorbar.locator = MaxNLocator(nbins=4)
    colorbar.update_ticks()
    plt.gca().xaxis.set_major_locator(MaxNLocator(4))
    plt.gca().yaxis.set_major_locator(MaxNLocator(4))

    for x0, y0, r in zip(x0_list, y0_list, r_list):
        circle = plt.Circle((x0, y0), r, color='white', fill=False, linestyle='--')
        plt.gca().add_patch(circle)

    grid_z1[mask] = 0  # Mask out the circle
    grid_z2[mask] = 0  # Mask out the circle
    rel_error = np.linalg.norm(grid_z2 - grid_z1) / np.linalg.norm(grid_z1)
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
    x_min, x_max = np.min(node1[:, 0]), np.max(node1[:, 0])
    y_min, y_max = np.min(node1[:, 1]), np.max(node1[:, 1])
    x_loc = x_min + 0.65 * (x_max - x_min)
    y_loc = y_min + 0.05 * (y_max - y_min)
    plt.text(x_loc, y_loc, f'Rel. $L^2$ error: {rel_error * 100:.1f} %', horizontalalignment='center', bbox=bbox_props, )
    plt.tight_layout()
    plt.title(title)
    plt.savefig(figname)
    if is_show:
        plt.show()
    plt.close()
    return rel_error

def get_don_results(net, trunk_input, branch_input):
    branch_input = torch.tensor(branch_input, dtype=torch.float32).view(1, -1)
    branch_input = branch_input.repeat(trunk_input.shape[0], 1)
    branch_input_min = torch.tensor(net.config["branch_input_min"])
    branch_input_max = torch.tensor(net.config["branch_input_max"])
    trunk_input_min = torch.tensor(net.config["trunk_input_min"])
    trunk_input_max = torch.tensor(net.config["trunk_input_max"])
    normalized_branch_input = (branch_input - branch_input_min) / (branch_input_max - branch_input_min)
    normalized_trunk_input = (trunk_input - trunk_input_min) / (trunk_input_max - trunk_input_min)
    input_tensor = torch.cat((normalized_branch_input, normalized_trunk_input), dim=1)

    # Neural network prediction
    net_output = net(input_tensor)

    output_min = torch.tensor(net.config["output_min"])
    output_max = torch.tensor(net.config["output_max"])
    results = net_output * (output_max - output_min) + output_min
    return results.detach().numpy()

def get_don_domain_results(net, x_len, y_len, T_BC, r, don_visual_node_num=100, x0=0.4, y0=0.4, is_zeromean=False):
    if is_zeromean:
        mean_T_BC = np.mean(T_BC)
        T_BC = np.array(T_BC, dtype=np.float32) - mean_T_BC
    else:
        mean_T_BC = 0
    don_visual_point = [(i + 0.5) / don_visual_node_num for i in range(don_visual_node_num)]
    don_visual_point = torch.tensor(don_visual_point)

    # Create 2D grid of points and weights
    gx, gy = torch.meshgrid(don_visual_point, don_visual_point, indexing='ij')

    # Generate and preprocess Gaussian points
    don_visual_point_x_input = (gx * x_len).flatten().unsqueeze(-1)
    don_visual_point_y_input = (gy * y_len).flatten().unsqueeze(-1)

    # Create mask
    mask = (don_visual_point_x_input - 0.5 * x_len) ** 2 + (don_visual_point_y_input - 0.5 * y_len) ** 2 < r ** 2
    mask = mask.flatten()

    # Filter out points within the circle
    don_visual_point_x_input_filtered = don_visual_point_x_input[~mask]
    don_visual_point_y_input_filtered = don_visual_point_y_input[~mask]

    don_visual_point = torch.cat((don_visual_point_x_input_filtered, don_visual_point_y_input_filtered), dim=1)
    results = get_don_results(net=net, trunk_input=don_visual_point, branch_input=T_BC)
    don_visual_point = don_visual_point.detach().numpy()
    don_visual_point[:, 0] = don_visual_point[:, 0] + x0
    don_visual_point[:, 1] = don_visual_point[:, 1] + y0
    return don_visual_point, results + mean_T_BC

if __name__ == "__main__":
    pass
