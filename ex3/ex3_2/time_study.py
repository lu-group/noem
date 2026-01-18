import gmsh, torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import ex3.fem_solver.fem_solver as fem_solver
import ex3.fem_solver.don_solver_v3 as don_solver
import ex3.ex3_2.get_fem_mesh as get_fem_mesh
import ex3.ex3_2.get_don_mesh as get_don_mesh

def get_fourside_nodeid(node_id, mesh_num_width, mesh_num_height):
    button_nodeid = node_id[:mesh_num_width + 1]
    right_nodeid = node_id[mesh_num_width: mesh_num_width + mesh_num_height + 1]
    top_nodeid = node_id[mesh_num_width + mesh_num_height: mesh_num_width * 2 + mesh_num_height + 1]
    left_nodeid = node_id[mesh_num_width * 2 + mesh_num_height: mesh_num_width * 2 + mesh_num_height * 2]
    left_nodeid = np.append(left_nodeid, node_id[0])
    return button_nodeid, right_nodeid, top_nodeid, left_nodeid

def run(left_T_BC, right_T_BC, outer_width, outer_height, hole_num_x, hole_num_y, net_name, results_name, results_path="compare_time_results"):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    # Generate the mesh
    outer_mesh_size = 0.1

    center_x_list = [(i + 0.5) / hole_num_x for i in range(hole_num_x)]
    center_x_list = np.array(center_x_list) * outer_width
    center_y_list = [(i + 0.5) / hole_num_y for i in range(hole_num_y)]
    center_y_list = np.array(center_y_list) * outer_height
    # Create the grid of points accoring to the center of the holes
    center_x_list, center_y_list = np.meshgrid(center_x_list, center_y_list)
    center_x_list = center_x_list.flatten()
    center_y_list = center_y_list.flatten()

    radius = 0.15
    radius_list = [radius for i in range(hole_num_x * hole_num_y)]
    mesh_size_circle = 0.04

    don_domain_hole_width = 0.8
    don_domain_hole_width_list = [don_domain_hole_width for i in range(hole_num_x * hole_num_y)]
    don_domain_hole_height = don_domain_hole_width
    don_domain_hole_height_list = [don_domain_hole_height for i in range(hole_num_x * hole_num_y)]
    don_domain_mesh_size_hole = 0.05

    from ex3.fem_solver.element import don_element as don_element
    is_branch_zeromean = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    don_element.DON_info.initialize(net_name=net_name, x_len=don_domain_hole_width, y_len=don_domain_hole_height,
                                    r=radius, is_zeromean=is_branch_zeromean, device=device)
    fem_time = 0
    don_time = 0

    import time
    start = time.time()
    fem_node, fem_element, fem_outline_indices = get_fem_mesh.get_mesh(mesh_size=outer_mesh_size, width=outer_width,
                                                                       height=outer_height,
                                                                       center_x_list=center_x_list,
                                                                       center_y_list=center_y_list,
                                                                       radius_list=radius_list,
                                                                       mesh_size_circle=mesh_size_circle)
    fem_time += time.time() - start

    start = time.time()
    don_node, don_element_mesh, don_outer_outline_indices, don_hole_outline_indices_list = get_don_mesh.generate_don_mesh(
        mesh_size=outer_mesh_size, outer_width=outer_width, outer_height=outer_height,
        hole_width_list=don_domain_hole_width_list, hole_height_list=don_domain_hole_height_list,
        hole_x_center_list=center_x_list, hole_y_center_list=center_y_list,
        mesh_size_hole=don_domain_mesh_size_hole)
    don_time += time.time() - start

    fem_buttom_nodeid, fem_right_nodeid, fem_top_nodeid, fem_left_nodeid = get_fourside_nodeid \
        (fem_outline_indices, int(outer_width / outer_mesh_size), int(outer_height / outer_mesh_size))
    don_buttom_nodeid, don_right_nodeid, don_top_nodeid, don_left_nodeid = get_fourside_nodeid \
        (don_outer_outline_indices, int(outer_width / outer_mesh_size), int(outer_height / outer_mesh_size))

    fem_T_BC = np.concatenate((left_T_BC, right_T_BC))
    fem_T_nodeid = np.concatenate((fem_left_nodeid, fem_right_nodeid))

    don_T_BC = fem_T_BC
    don_T_nodeid = np.concatenate((don_left_nodeid, don_right_nodeid))

    q_nodeid = np.array([])
    q_mag = 0
    q_BC = np.zeros(len(q_nodeid))

    startT = time.time()
    fem_T = fem_solver.solver(node=fem_node, mesh=fem_element, T_BC=fem_T_BC, T_nodeid=fem_T_nodeid, q_BC=q_BC,
                              q_nodeid=q_nodeid,
                              k=1, element_type='tri3')
    fem_time += time.time() - startT
    # print("FEM Simulation Done")
    startT = time.time()
    don_T, ele_cal_time = don_solver.one_solver(node=don_node, mesh=don_element_mesh, T_BC=don_T_BC, T_nodeid=don_T_nodeid,
                                  q_BC=q_BC, q_nodeid=q_nodeid, T_oneBC_nodeid_list=don_hole_outline_indices_list,
                                  don_element=don_element,
                                  k=1, element_type='tri3')
    # print("DON Simulation Done")
    don_time += time.time() - startT

    import ex3.ex3_2.visualization as visualization
    fem_results_fig_name = os.path.join(results_path, "fem_results.png")
    visualization.plot_2d_contour(node=fem_node, results=fem_T,
                                  x0_list=center_x_list, y0_list=center_y_list, r_list=radius_list, title="FEM Results",
                                  figname=fem_results_fig_name, is_show=False)

    for i in range(len(don_hole_outline_indices_list)):
        don_hole_outline_indices = don_hole_outline_indices_list[i]
        don_domain_hole_width = don_domain_hole_width_list[i]
        don_domain_hole_height = don_domain_hole_height_list[i]
        x0 = center_x_list[i] - don_domain_hole_width / 2
        y0 = center_y_list[i] - don_domain_hole_height / 2
        don_T_oneBC = don_T[don_hole_outline_indices]
        don_visual_points, don_visual_T = visualization.get_don_domain_results(net=don_element.DON_info.net.to("cpu"),
                                                                               x_len=don_domain_hole_width,
                                                                               y_len=don_domain_hole_height,
                                                                               T_BC=don_T_oneBC, r=radius,
                                                                               don_visual_node_num=100, x0=x0, y0=y0,
                                                                               is_zeromean=is_branch_zeromean)
        don_visual_points = np.append(don_node, don_visual_points, axis=0)
        don_visual_T = np.append(don_T, don_visual_T)

    don_results_fig_name = os.path.join(results_path, "don_results.png")
    visualization.plot_2d_contour(node=don_visual_points, results=don_visual_T,
                                  x0_list=center_x_list, y0_list=center_y_list, r_list=radius_list, title="DON Results",
                                  figname=don_results_fig_name, is_show=False)

    diff_results_fig_name = os.path.join(results_path, "diff_results.png")
    rel_error = visualization.plot_2d_diff_contour(node1=fem_node, results1=fem_T, node2=don_visual_points, results2=don_visual_T,
                                       x0_list=center_x_list, y0_list=center_y_list, r_list=radius_list,
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
    plt.savefig(os.path.join(results_path, results_name))
    plt.close()
    return fem_time, don_time, ele_cal_time, rel_error
from tqdm import *
def compare_time():
    block_width = block_height = 1.5
    hole_num_list = np.arange(1, 11)
    net_name = "..//data_driven_training//ex3//ex3.pth"
    sample_num = 20
    results_sum = {}
    FEM_T = []
    DON_T = []
    for hole_num in hole_num_list:
        hole_num_x = hole_num
        hole_num_y = hole_num
        outer_width = hole_num_x * block_width
        outer_height = hole_num_y * block_height
        results_sum[(hole_num_x, hole_num_y)] = []
        desc = "Simulation for plates with hole_num_x=" + str(hole_num_x) + ", hole_num_y=" + str(hole_num_y) + "..."
        pbar = tqdm(range(sample_num), desc=desc)
        left_T_BC_list = np.load("ex3_2//bc_set//HoleNum_" + str(hole_num_x) + "BCs.npz")["left_T_BC_list"]
        right_T_BC_list = np.load("ex3_2//bc_set//HoleNum_" + str(hole_num_x) + "BCs.npz")["right_T_BC_list"]
        tFEM_T = 0
        tDON_T = 0
        for i in pbar:
            left_T_BC = left_T_BC_list[i]
            right_T_BC = right_T_BC_list[i]
            tresults_name = "results_" + str(hole_num_x) + "_" + str(hole_num_y) + "_" + str(i) + ".png"
            fem_time, don_time, _, rel_error = run(left_T_BC=left_T_BC, right_T_BC=right_T_BC, outer_width=outer_width,
                                                   outer_height=outer_height,
                                                   hole_num_x=hole_num_x, hole_num_y=hole_num_y,
                                                   net_name=net_name, results_name=tresults_name)
            results_sum[(hole_num_x, hole_num_y)].append(rel_error)
            tFEM_T += fem_time
            tDON_T += don_time
        FEM_T.append(tFEM_T / sample_num)
        DON_T.append(tDON_T / sample_num)
        print("Results for hole_num_x=" + str(hole_num_x) + ", hole_num_y=" + str(hole_num_y) + ":")
        print(results_sum[(hole_num_x, hole_num_y)])

    print()
    print("Computational Cost")
    print("Hole Num, FEM, DON")
    for i in range(len(hole_num_list)):
        print(hole_num_list[i], round(FEM_T[i], 2), round(DON_T[i], 2))
if __name__ == "__main__":
    compare_time()
