import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from matplotlib.ticker import MaxNLocator

colormap = "rainbow"
fontsize = 12
plt.rcParams['font.size'] = fontsize
figsize = (5, 4.05)

def visual_mesh(node, element):
    plt.figure()

    for tri in element:
        triangle = plt.Polygon(node[tri, :2], edgecolor='black', facecolor='none')
        plt.gca().add_patch(triangle)

    plt.scatter(node[:, 0], node[:, 1], c='red', marker='o', s=0.5)  # Node points
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Mesh Visualization')
    plt.grid(True)
    # plt.legend()
    plt.show()

def plot_2d_contour(node, results, x0_list, y0_list, r_list, vmin=None, vmax=None, figname="contour.png",
                    node_visual=False, is_show=True, title=""):
    if type(node) == list:
        node = np.array(node)
    x_min, x_max = np.min(node[:, 0]), np.max(node[:, 0])
    y_min, y_max = np.min(node[:, 1]), np.max(node[:, 1])
    grid_x, grid_y = np.mgrid[
                     x_min:x_max:300j,  # Increase resolution for smoother edges
                     y_min:y_max:300j
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
    plt.title(None)
    plt.legend(loc='best') if node_visual else None
    plt.savefig(figname)
    if is_show:
        plt.show()
    plt.close()

def plot_2d_diff_contour2(node1, results1, node2, results2, x0_list, y0_list, r_list,
                         vmin=None, vmax=None, figname="contour_diff.png", title="", is_show=True):
    grid_x, grid_y = np.mgrid[
                     np.min(node1[:, 0]):np.max(node1[:, 0]):300j,  # Increase resolution for smoother edges
                     np.min(node1[:, 1]):np.max(node1[:, 1]):300j
                     ]
    grid = np.array([grid_x.ravel(), grid_y.ravel()]).T
    grid_z1 = griddata(node1, results1.ravel(), (grid_x, grid_y), method='linear')
    grid_z2 = griddata(node2, results2.ravel(), (grid_x, grid_y), method='linear')
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

    plt.title(None)
    plt.savefig(figname)
    if is_show:
        plt.show()
    plt.close()
    return grid, grid_z

def plot_2d_diff_contour(node1, results1, node2, results2, x0_list, y0_list, r_list,
                         vmin=None, vmax=None, figname="contour_diff.png", is_show=True):
    grid_x, grid_y = np.mgrid[
                     np.min(node1[:, 0]):np.max(node1[:, 0]):20j,  # Increase resolution for smoother edges
                     np.min(node1[:, 1]):np.max(node1[:, 1]):20j
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
    x_loc = x_min + 0.6 * (x_max - x_min)
    y_loc = y_min + 0.05 * (y_max - y_min)
    plt.text(x_loc, y_loc, f'Rel. $L^2$ error: {rel_error*100:.1f}%', horizontalalignment='center', bbox=bbox_props, fontsize=fontsize+2)

    plt.title(None)
    plt.savefig(figname)
    if is_show:
        plt.show()
    plt.close()
    return rel_error

def get_don_results(net, trunk_input, branch_input):
    branch_input = torch.tensor(branch_input, dtype=torch.float32).view(1, -1)
    branch_input_min = torch.tensor(net.config["branch_input_min"])
    branch_input_max = torch.tensor(net.config["branch_input_max"])
    trunk_input_min = torch.tensor(net.config["trunk_input_min"])
    trunk_input_max = torch.tensor(net.config["trunk_input_max"])
    trunk_input = torch.tensor(trunk_input, dtype=torch.float32).view(1, -1)
    normalized_branch_input = (branch_input - branch_input_min) / (branch_input_max - branch_input_min)
    normalized_trunk_input = (trunk_input - trunk_input_min) / (trunk_input_max - trunk_input_min)
    input_tensor = torch.cat((normalized_branch_input, normalized_trunk_input), dim=1)

    net_output = net(input_tensor)

    output_min = torch.tensor(net.config["output_min"])
    output_max = torch.tensor(net.config["output_max"])
    results = net_output * (output_max - output_min) + output_min
    return results.detach().numpy()

if __name__ == "__main__":
    pass
