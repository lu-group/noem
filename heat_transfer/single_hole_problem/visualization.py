import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from matplotlib.ticker import MaxNLocator

colormap = "rainbow"
fontsize = 8
plt.rcParams['font.size'] = fontsize
figsize = (3.0,2.4)
def visual_mesh(node, element, scatter_size=1):
    plt.figure()
    for tri in element:
        triangle = plt.Polygon(node[tri, :2], edgecolor='black', facecolor='none')
        plt.gca().add_patch(triangle)

    plt.scatter(node[:, 0], node[:, 1], c='red', marker='o', s=scatter_size)  # Node points
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_2d_contour(node, results, x0, y0, r, vmin=None, vmax=None, figname="contour.png", title=None, node_visual=False, is_show=True):
    grid_x, grid_y = np.mgrid[
                     0.04:1.6 - 0.04:500j,  # Increase resolution for smoother edges
                     0.04:1.6 - 0.04:500j
                     ]

    grid_z = griddata(node, results.ravel(), (grid_x, grid_y), method='linear')

    mask = (grid_x - x0) ** 2 + (grid_y - y0) ** 2 <= r ** 2
    grid_z[mask] = np.nan  # Mask out the circle

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
    circle = plt.Circle((x0, y0), r, color='white', fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    if node_visual:
        # Plot nodes for reference
        plt.scatter(node[:, 0], node[:, 1], c='red', s=5, label='Nodes')
        plt.scatter([x0], [y0], c='blue', s=50, label='Circle Center')

    plt.title(title)
    plt.tight_layout()
    plt.legend(loc='best') if node_visual else None
    if title is not None:
        plt.title(title)
    plt.savefig(figname)
    if is_show:
        plt.show()


def plot_2d_diff_contour(node1, results1, node2, results2, x0, y0, r,  title=None,
                         vmin=None, vmax=None, figname="contour_diff.png", node_visual=False, is_show=True):

    grid_x, grid_y = np.mgrid[
                     0.04:1.6 - 0.04:500j,  # Increase resolution for smoother edges
                     0.04:1.6 - 0.04:500j
                     ]

    grid_z1 = griddata(node1, results1.ravel(), (grid_x, grid_y), method='linear')
    grid_z2 = griddata(node2, results2.ravel(), (grid_x, grid_y), method='linear')
    grid_z = np.abs(grid_z1 - grid_z2)

    mask = (grid_x - x0) ** 2 + (grid_y - y0) ** 2 <= r ** 2
    grid_z[mask] = np.nan

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

    circle = plt.Circle((x0, y0), r, color='white', fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    grid_z1[mask] = 0  # Mask out the circle
    grid_z2[mask] = 0  # Mask out the circle
    rel_error = np.linalg.norm(grid_z2 - grid_z1) / np.linalg.norm(grid_z1)

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
    x_min, x_max = np.min(node1[:, 0]), np.max(node1[:, 0])
    y_min, y_max = np.min(node1[:, 1]), np.max(node1[:, 1])
    x_loc = 0.85
    y_loc = y_min + 0.05 * (y_max - y_min)
    plt.text(x_loc, y_loc, f'Rel. $L^2$ error: {rel_error * 100:.1f} %', horizontalalignment='center', bbox=bbox_props, fontsize=fontsize+2.5)

    plt.title(title)
    plt.legend(loc='best') if node_visual else None
    plt.savefig(figname)
    if is_show:
        plt.show()

def plot_2d_diff_contour2(node1, results1, node2, results2, x0_list, y0_list, r_list,
                         vmin=None, vmax=None, figname="contour_diff.png", title="", is_show=True):
    grid_x, grid_y = np.mgrid[
                     0.04:1.6 - 0.04:500j,  # Increase resolution for smoother edges
                     0.04:1.6 - 0.04:500j
                     ]
    grid = np.array([grid_x.ravel(), grid_y.ravel()]).T
    grid_z1 = griddata(node1, results1.ravel(), (grid_x, grid_y), method='linear')
    grid_z2 = griddata(node2, results2.ravel(), (grid_x, grid_y), method='linear')
    grid_z1[np.isnan(grid_z1)] = 0
    grid_z2[np.isnan(grid_z2)] = 0
    grid_z = grid_z2

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
    grid_z[mask] = 0  # Mask out the circle

    plt.title(title)
    plt.tight_layout()

    plt.savefig(figname)
    if is_show:
        plt.show()
    plt.close()
    return grid, grid_z1, grid_z

def plot_2d_diff_contourv2(node1, results1, node2, results2, x0, y0, r,
                         vmin=None, vmax=None, figname="contour_diff.png", node_visual=False, is_show=True):

    grid_x, grid_y = np.mgrid[
                     0.1:1.6-0.1:500j,  # Increase resolution for smoother edges
                     0.1:1.6-0.1:500j
                     ]
    grid_z1 = griddata(node1, results1.ravel(), (grid_x, grid_y), method='linear')
    grid_z2 = griddata(node2, results2.ravel(), (grid_x, grid_y), method='linear')
    # Transfer all nan to 0
    grid_z1[np.isnan(grid_z1)] = 0
    grid_z2[np.isnan(grid_z2)] = 0
    grid_z = np.abs(grid_z1 - grid_z2)

    mask = (grid_x - x0) ** 2 + (grid_y - y0) ** 2 <= r ** 2
    grid_z[mask] = np.nan  # Mask out the circle

    # Plot the contour
    plt.figure(figsize=figsize)
    levels = 100
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=colormap, vmin=vmin, vmax=vmax)

    # Adjust color bar based on vmin and vmax
    if vmin is None or vmax is None:
        vmin, vmax = np.nanmin(grid_z1), np.nanmax(grid_z1)

    colorbar = plt.colorbar(contour, boundaries=np.linspace(vmin, vmax, levels))
    colorbar.locator = MaxNLocator(nbins=4)
    colorbar.update_ticks()
    plt.gca().xaxis.set_major_locator(MaxNLocator(4))
    plt.gca().yaxis.set_major_locator(MaxNLocator(4))
    # Plot the circle boundary
    circle = plt.Circle((x0, y0), r, color='white', fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    grid_z1[mask] = 0  # Mask out the circle
    grid_z2[mask] = 0  # Mask out the circle
    grid_z[mask] = 0  # Mask out the circle
    # Calculate the relative L2 error against the reference solution grid_z1
    rel_error = np.linalg.norm(grid_z) / np.linalg.norm(grid_z1)
    print("rel_error", rel_error)
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
    x_min, x_max = np.min(node1[:, 0]), np.max(node1[:, 0])
    y_min, y_max = np.min(node1[:, 1]), np.max(node1[:, 1])
    x_loc = 0.85
    y_loc = y_min + 0.1 * (y_max - y_min)
    plt.text(x_loc, y_loc, f'Rel. $L^2$ error: {rel_error * 100:.1f} %', horizontalalignment='center', bbox=bbox_props, fontsize=fontsize+2.5)
    plt.tight_layout()

    plt.legend(loc='best') if node_visual else None
    plt.savefig(figname)
    if is_show:
        plt.show()
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

def get_don_resultsv2(net, trunk_input, branch_input):
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
    return results
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

def get_don_domain_results_T_dx(net, x_len, y_len, T_BC, r, don_visual_node_num=100, x0=0.4, y0=0.4, is_zeromean=False):
    if is_zeromean:
        mean_T_BC = np.mean(T_BC)
        T_BC = np.array(T_BC, dtype=np.float32) - mean_T_BC
    else:
        mean_T_BC = 0
    don_visual_point = [(i + 0.5) / don_visual_node_num for i in range(don_visual_node_num)]
    don_visual_point = torch.tensor(don_visual_point)

    gx, gy = torch.meshgrid(don_visual_point, don_visual_point, indexing='ij')

    don_visual_point_x_input = (gx * x_len).flatten().unsqueeze(-1)
    don_visual_point_y_input = (gy * y_len).flatten().unsqueeze(-1)

    # Create mask
    mask = (don_visual_point_x_input - 0.5 * x_len) ** 2 + (don_visual_point_y_input - 0.5 * y_len) ** 2 < r ** 2
    mask = mask.flatten()

    # Filter out points within the circle
    don_visual_point_x_input_filtered = don_visual_point_x_input[~mask]
    don_visual_point_y_input_filtered = don_visual_point_y_input[~mask]

    don_visual_point_x_input_filtered = don_visual_point_x_input_filtered.detach().requires_grad_(True)
    don_visual_point_y_input_filtered = don_visual_point_y_input_filtered.detach().requires_grad_(True)

    don_visual_point = torch.cat((don_visual_point_x_input_filtered, don_visual_point_y_input_filtered), dim=1)
    results = get_don_resultsv2(net=net, trunk_input=don_visual_point, branch_input=T_BC)
    T_dx = torch.autograd.grad(results.sum(), don_visual_point_x_input_filtered, create_graph=True)[0]
    T_dx = T_dx.detach().numpy()
    # don_visual_point = don_visual_point.detach().numpy()
    don_visual_point[:, 0] = don_visual_point[:, 0] + x0
    don_visual_point[:, 1] = don_visual_point[:, 1] + y0
    # return don_visual_point, results + mean_T_BC
    return don_visual_point.detach().numpy(), T_dx

if __name__ == "__main__":
    pass
