import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.util.grf import grf_1D
import ex4.ex4_4.visualization as visualization
import os, torch
import ex4.fem_solver_ex4_4.fem_solver as fem_solver
from ex4.fem_solver_ex4_4.don_solver import one_solver

BASE_DIR = Path(__file__).resolve().parent
def transfer_nodeid(old_nodeid, existed_orignal_nodeid, existed_new_nodeid):
    idx = existed_orignal_nodeid.index(old_nodeid)
    return existed_new_nodeid[idx]

def get_eleloc(node, mesh):
    node = np.array(node)
    mesh = np.array(mesh)
    ele_loc = []
    for ele in mesh:
        ele_coords = node[ele]
        ele_center = np.mean(ele_coords, axis=0)
        ele_loc.append(ele_center)
    return np.array(ele_loc)

def del_nodo(node, mesh, x_start, x_end, y_start, y_end):
    new_node = {}
    for i in range(len(node)):
        x, y = node[i]
        if (x > x_start and x < x_end) and (y > y_start and y < y_end):
            continue
        new_node[i] = [x, y]
    existed_node = []
    existed_orignal_nodeid = []
    existed_new_nodeid = []
    counter = 0
    for key, value in new_node.items():
        existed_node.append(value)
        existed_orignal_nodeid.append(key)
        existed_new_nodeid.append(counter)
        counter += 1

    new_mesh = []
    for nodes in mesh:
        del_ele = False
        for i in nodes:
            if i not in existed_orignal_nodeid:
                del_ele = True
        if del_ele:
            continue
        else:
            new_nodes = [transfer_nodeid(i, existed_orignal_nodeid, existed_new_nodeid) for i in nodes]
            new_mesh.append(new_nodes)
    return existed_node, new_mesh, existed_orignal_nodeid, existed_new_nodeid

def get_oneBC_nodeid(num_x, num_y):
    start_interval = int(0.5 / 2 * num_x)
    mesh_size = int(num_x / 2.0)
    left_nodeid = [(i + start_interval) * (num_x + 1) + start_interval for i in range(int(mesh_size * 1 + 1))]
    right_nodeid = [(i + start_interval) * (num_x + 1) + (num_y - start_interval) for i in range(int(mesh_size * 1 + 1))]
    top_nodeid = [i + start_interval + (num_x + 1) * (num_y - start_interval) for i in range(int(mesh_size * 1 + 1))]
    bottom_nodeid = [i + start_interval + (num_x + 1) * start_interval for i in range(int(mesh_size * 1 + 1))]
    ToneBC_nodeid = bottom_nodeid + right_nodeid[1::] + top_nodeid[-2::-1] + left_nodeid[-2::-1]
    ToneBC_nodeid = ToneBC_nodeid[:-1]
    return ToneBC_nodeid

def get_K():
    x_len = 1.0
    y_len = 1.0
    num_x = num_y = 20
    fem_node = fem_solver.get_node(x_len, y_len, num_x, num_y)
    fem_mesh = fem_solver.get_mesh(num_x, num_y)
    ele_loc = get_eleloc(fem_node, fem_mesh)
    randon_field_K = np.zeros(len(ele_loc))
    return randon_field_K, ele_loc

def get_k_list(fem_ele_loc, randon_field_K, don_ele_loc):
    k_list = []
    don_ele_loc[:, 0] = don_ele_loc[:, 0] + 0.5
    don_ele_loc[:, 1] = don_ele_loc[:, 1] + 0.5
    for i in range(len(fem_ele_loc)):
        tele_loc = fem_ele_loc[i]
        # Check whether tele_loc is in the don_ele_loc
        if np.any(np.all(don_ele_loc == tele_loc, axis=1)):
            idx = np.where(np.all(don_ele_loc == tele_loc, axis=1))[0][0]
            k_list.append(randon_field_K[idx])
        else:
            k_list.append(1)
    return k_list

def run(net_name="..//data_driven_training//ex4_darcy//ex4_4.pth",
        results_name=None):
    x_len = y_len = 2.0
    num_x = num_y = 40
    node = fem_solver.get_node(x_len, y_len, num_x, num_y)
    mesh = fem_solver.get_mesh(num_x, num_y)

    # Define boundary conditions
    left_nodeid = [i * (num_x + 1) for i in range(num_y + 1)]
    right_nodeid = [(i + 1) * (num_x + 1) - 1 for i in range(num_y + 1)]
    top_nodeid = [i + (num_x + 1) * num_y for i in range(num_x + 1)]
    bottom_nodeid = [i for i in range(num_x + 1)]
    outline_nodes = bottom_nodeid[:-1] + right_nodeid[:-1] + top_nodeid[:-1] + left_nodeid[:-1]

    T_nodeid = left_nodeid
    T_nodeid += right_nodeid
    bc_path = BASE_DIR / "boundary_conditions_ex_4_4.npz"
    left_T_BC = np.load(bc_path)['left_T_BC']
    right_T_BC = np.load(bc_path)['right_T_BC']
    T_BC = np.concatenate((left_T_BC, right_T_BC))

    q_nodeid = np.concatenate((top_nodeid, bottom_nodeid))
    q_mag = 0
    q_BC = np.ones(len(q_nodeid)) * q_mag * x_len / num_x  # Evenly distribute heat flux

    T_oneBC_nodeid = get_oneBC_nodeid(num_x, num_y)

    don_node, don_mesh, existed_orignal_nodeid, existed_new_nodeid = del_nodo(node, mesh, 0.5, 1.5, 0.5, 1.5)
    don_q_nodeid = np.zeros_like(q_nodeid)
    for i in range(len(T_nodeid)):
        T_nodeid[i] = transfer_nodeid(T_nodeid[i], existed_orignal_nodeid, existed_new_nodeid)
    for i in range(len(q_nodeid)):
        don_q_nodeid[i] = transfer_nodeid(q_nodeid[i], existed_orignal_nodeid, existed_new_nodeid)
    for i in range(len(T_oneBC_nodeid)):
        T_oneBC_nodeid[i] = transfer_nodeid(T_oneBC_nodeid[i], existed_orignal_nodeid, existed_new_nodeid)

    randon_field_K, don_ele_loc = get_K()
    # net_name = "..//..//data_driven_training//ex4_darcy//ex4.2_darcy_20mesh_5000sam0622.pth"
    from ex4.fem_solver_ex4_4.element import don_element_ex4 as don_element
    don_element.DON_info.initialize(net_name=net_name, x_len=1, y_len=1, r=0)
    one_T = one_solver(node=don_node, mesh=don_mesh, T_BC=T_BC, T_nodeid=T_nodeid, q_BC=q_BC, q_nodeid=don_q_nodeid,
                       T_oneBC_nodeid_list=[T_oneBC_nodeid], don_element=don_element, k_fem_list=np.ones(len(don_mesh)), element_type='qua4')
    fem_node = fem_solver.get_node(x_len, y_len, num_x, num_y)
    fem_mesh = fem_solver.get_mesh(num_x, num_y)
    left_nodeid = [i * (num_x + 1) for i in range(num_y + 1)]
    right_nodeid = [(i + 1) * (num_x + 1) - 1 for i in range(num_y + 1)]
    top_nodeid = [i + (num_x + 1) * num_y for i in range(num_x + 1)]
    bottom_nodeid = [i for i in range(num_x + 1)]

    fem_T_BC = T_BC
    fem_T_nodeid = np.concatenate((left_nodeid, right_nodeid))
    fem_ele_loc = get_eleloc(fem_node, fem_mesh)
    fem_k_list = get_k_list(fem_ele_loc, randon_field_K, don_ele_loc)
    fem_T = fem_solver.solver(node=fem_node, mesh=fem_mesh, T_BC=fem_T_BC, T_nodeid=fem_T_nodeid, q_BC=q_BC, q_nodeid=q_nodeid,
                              k_list=fem_k_list, element_type='quad4')

    results_path = BASE_DIR / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    if results_name is None:
        results_name = results_path / "combined results.png"
    else:
        results_name = Path(results_name)
        if not results_name.is_absolute():
            results_name = results_path / results_name

    fem_results_fig_name = os.path.join(results_path, "fem_results.png")
    visualization.plot_2d_contour(node=np.array(fem_node), results=np.array(fem_T),
                                  x0_list=[], y0_list=[], r_list=[], title="FEM Results",
                                  figname=fem_results_fig_name, is_show=False)

    grid_x, grid_y = np.meshgrid(np.linspace(0.5 * 1 / 20, (20 - 0.5) / 20 * 1, 20 - 1),
                                 np.linspace(0.5 * 1 / 20, (20 - 0.5) / 20 * 1, 20 - 1))
    don_visual_points = np.array([grid_x.flatten(), grid_y.flatten()]).T
    don_visual_T = np.zeros(len(don_visual_points))
    don_BC_T = one_T[T_oneBC_nodeid]
    for i in range(len(don_visual_points)):
        don_visual_T[[i]] = visualization.get_don_results(net=don_element.DON_info.net,
                                                          trunk_input=torch.tensor(don_visual_points[i]), branch_input=don_BC_T)
    don_visual_points[:, 0] = don_visual_points[:, 0] + 0.5
    don_visual_points[:, 1] = don_visual_points[:, 1] + 0.5
    don_visual_points = np.concatenate((don_visual_points, don_node))
    don_visual_T = np.concatenate((don_visual_T, one_T))

    don_results_fig_name = os.path.join(results_path, "don_results.png")
    don_visual_points, don_visual_T = visualization.plot_2d_diff_contour2(node1=np.array(fem_node), results1=fem_T,
                                                                          node2=don_visual_points,
                                                                          results2=don_visual_T,
                                                                          x0_list=[], y0_list=[], r_list=[],
                                                                          title="NOE Results",
                                                                          figname=don_results_fig_name, is_show=False)

    diff_results_fig_name = os.path.join(results_path, "diff_results.png")
    visualization.plot_2d_diff_contour(node1=np.array(fem_node), results1=fem_T,
                                       node2=don_visual_points, results2=don_visual_T,
                                       x0_list=[], y0_list=[], r_list=[],
                                       figname=diff_results_fig_name, is_show=False)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Create a 1x3 grid of subplots

    # Load and display the first figure
    img1 = plt.imread(fem_results_fig_name)
    axs[0].imshow(img1)
    axs[0].axis('off')

    # Load and display the second figure
    img2 = plt.imread(don_results_fig_name)
    axs[1].imshow(img2)
    axs[1].axis('off')

    # Load and display the third figure
    img3 = plt.imread(diff_results_fig_name)
    axs[2].imshow(img3)
    axs[2].axis('off')

    # Adjust layout to fit subplots into the figure area
    plt.tight_layout()
    plt.savefig(results_name)
    plt.show()

