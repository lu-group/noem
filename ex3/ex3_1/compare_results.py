import gmsh, torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import ex3.fem_solver.fem_solver as fem_solver
import ex3.fem_solver.don_solver as don_solver

def get_fem_model(mesh_size, width, height, radius, mesh_size_circle=None):
    from ex3.fem_solver import meshing
    return meshing.get_mesh(mesh_size=mesh_size, width=width, height=height, radius=radius, mesh_size_circle=mesh_size_circle)

def get_fourside_nodeid(node_id, mesh_num):
    button_nodeid = node_id[:mesh_num + 1]
    right_nodeid = node_id[mesh_num: mesh_num *2 + 1]
    top_nodeid = node_id[mesh_num * 2: mesh_num * 3 + 1]
    left_nodeid = node_id[mesh_num * 3: mesh_num * 4]
    left_nodeid = np.append(left_nodeid, node_id[0])
    return button_nodeid, right_nodeid, top_nodeid, left_nodeid

def run_bc1():
    # Generate the mesh
    outer_width = 1.6
    outer_height = outer_width
    outer_mesh_size = 0.1
    mesh_size = 0.1
    radius = 0.15
    mesh_size_circle = 0.01

    don_domain_hole_width = 0.8
    don_domain_hole_height = don_domain_hole_width
    don_domain_hole_x_center = outer_width / 2
    don_domain_hole_y_center = outer_height / 2
    don_domain_mesh_size_hole = 0.05

    net_name = "..//data_driven_training//ex3//ex3.pth"
    from ex3.fem_solver.element import don_element as don_element
    is_branch_zeromean = True
    don_element.DON_info.initialize(net_name=net_name, x_len=don_domain_hole_width, y_len=don_domain_hole_height,
                                    r=radius, is_zeromean=is_branch_zeromean)

    fem_node, fem_element, fem_outline_indices = get_fem_model(mesh_size=mesh_size, width=outer_width,
                                                               height=outer_height, radius=radius,
                                                               mesh_size_circle=mesh_size_circle)

    import ex3.ex3_1.get_don_mesh as get_don_mesh
    don_node, don_element_mesh, don_outer_outline_indices, don_hole_outline_indices = get_don_mesh.generate_mesh(
        mesh_size=outer_mesh_size, outer_width=outer_width, outer_height=outer_height,
        hole_width=don_domain_hole_width, hole_height=don_domain_hole_height,
        hole_x_center=don_domain_hole_x_center, hole_y_center=don_domain_hole_y_center,
        mesh_size_hole=don_domain_mesh_size_hole)

    fem_buttom_nodeid, fem_right_nodeid, fem_top_nodeid, fem_left_nodeid = get_fourside_nodeid \
        (fem_outline_indices, int(outer_width / mesh_size))
    don_buttom_nodeid, don_right_nodeid, don_top_nodeid, don_left_nodeid = get_fourside_nodeid \
        (don_outer_outline_indices, int(outer_width / mesh_size))

    left_T_BC = np.load("ex3_1//bc1.npz")['left_T_BC']
    right_T_BC = np.load("ex3_1//bc1.npz")['right_T_BC']

    fem_T_BC = np.concatenate((left_T_BC, right_T_BC))
    fem_T_nodeid = np.concatenate((fem_left_nodeid, fem_right_nodeid))

    don_T_BC = fem_T_BC
    don_T_nodeid = np.concatenate((don_left_nodeid, don_right_nodeid))

    q_nodeid = np.array([])
    q_BC = np.zeros(len(q_nodeid))

    fem_T = fem_solver.solver(node=fem_node, mesh=fem_element, T_BC=fem_T_BC, T_nodeid=fem_T_nodeid, q_BC=q_BC,
                              q_nodeid=q_nodeid,
                              k=1, element_type='tri3')
    print("FEM Simulation Done")

    don_T = don_solver.one_solver(node=don_node, mesh=don_element_mesh, T_BC=don_T_BC, T_nodeid=don_T_nodeid,
                                  q_BC=q_BC, q_nodeid=q_nodeid, T_oneBC_nodeid=don_hole_outline_indices,
                                  don_element=don_element,
                                  k=1, element_type='tri3')
    print("DON Simulation Done")

    import ex3.ex3_1.visualization as visualization
    fem_results_fig_name = "ex3_1//results//fem_results.png"
    visualization.plot_2d_contour(node=fem_node, results=fem_T,
                                  x0=don_domain_hole_x_center, y0=don_domain_hole_y_center, r=radius,
                                  figname=fem_results_fig_name, is_show=False)

    T_oneBC = don_T[don_hole_outline_indices]
    don_visual_points, don_visual_T = visualization.get_don_domain_results(net=don_element.DON_info.net,
                                                                           x_len=don_domain_hole_width,
                                                                           y_len=don_domain_hole_height,
                                                                           T_BC=T_oneBC, r=radius,
                                                                           don_visual_node_num=100, x0=0.4, y0=0.4,
                                                                           is_zeromean=is_branch_zeromean)
    don_visual_points = np.append(don_node, don_visual_points, axis=0)
    don_visual_T = np.append(don_T, don_visual_T)
    don_results_fig_name = "ex3_1//results//don_results.png"
    visualization.plot_2d_contour(node=don_visual_points, results=don_visual_T,
                                  x0=don_domain_hole_x_center, y0=don_domain_hole_y_center, r=radius,
                                  figname=don_results_fig_name, is_show=False)

    # Visualization of the difference
    diff_results_fig_name = "ex3_1//results//diff_results.png"
    visualization.plot_2d_diff_contourv2(node1=fem_node, results1=fem_T, node2=don_visual_points, results2=don_visual_T,
                                         x0=don_domain_hole_x_center, y0=don_domain_hole_y_center, r=radius,
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
    plt.savefig("ex3_1//results//combined results.png")
    plt.show()


def run_bc2():
    # Generate the mesh
    outer_width = 1.6
    outer_height = outer_width
    outer_mesh_size = 0.1
    mesh_size = 0.1
    radius = 0.15
    mesh_size_circle = 0.01

    don_domain_hole_width = 0.8
    don_domain_hole_height = don_domain_hole_width
    don_domain_hole_x_center = outer_width / 2
    don_domain_hole_y_center = outer_height / 2
    don_domain_mesh_size_hole = 0.05

    net_name = "..//data_driven_training//ex3//ex3.pth"
    from ex3.fem_solver.element import don_element as don_element
    is_branch_zeromean = True
    don_element.DON_info.initialize(net_name=net_name, x_len=don_domain_hole_width, y_len=don_domain_hole_height,
                                    r=radius, is_zeromean=is_branch_zeromean)

    fem_node, fem_element, fem_outline_indices = get_fem_model(mesh_size=mesh_size, width=outer_width,
                                                               height=outer_height, radius=radius,
                                                               mesh_size_circle=mesh_size_circle)

    import ex3.ex3_1.get_don_mesh as get_don_mesh
    don_node, don_element_mesh, don_outer_outline_indices, don_hole_outline_indices = get_don_mesh.generate_mesh(
        mesh_size=outer_mesh_size, outer_width=outer_width, outer_height=outer_height,
        hole_width=don_domain_hole_width, hole_height=don_domain_hole_height,
        hole_x_center=don_domain_hole_x_center, hole_y_center=don_domain_hole_y_center,
        mesh_size_hole=don_domain_mesh_size_hole)

    fem_buttom_nodeid, fem_right_nodeid, fem_top_nodeid, fem_left_nodeid = get_fourside_nodeid \
        (fem_outline_indices, int(outer_width / mesh_size))
    don_buttom_nodeid, don_right_nodeid, don_top_nodeid, don_left_nodeid = get_fourside_nodeid \
        (don_outer_outline_indices, int(outer_width / mesh_size))

    left_T_BC = np.load("ex3_1//bc2.npz")['left_T_BC']
    right_T_BC = np.load("ex3_1//bc2.npz")['right_T_BC']

    fem_T_BC = np.concatenate((left_T_BC, right_T_BC))
    fem_T_nodeid = np.concatenate((fem_left_nodeid, fem_right_nodeid))

    don_T_BC = fem_T_BC
    don_T_nodeid = np.concatenate((don_left_nodeid, don_right_nodeid))

    q_nodeid = np.array([])
    q_BC = np.zeros(len(q_nodeid))

    fem_T = fem_solver.solver(node=fem_node, mesh=fem_element, T_BC=fem_T_BC, T_nodeid=fem_T_nodeid, q_BC=q_BC,
                              q_nodeid=q_nodeid,
                              k=1, element_type='tri3')
    print("FEM Simulation Done")

    don_T = don_solver.one_solver(node=don_node, mesh=don_element_mesh, T_BC=don_T_BC, T_nodeid=don_T_nodeid,
                                  q_BC=q_BC, q_nodeid=q_nodeid, T_oneBC_nodeid=don_hole_outline_indices,
                                  don_element=don_element,
                                  k=1, element_type='tri3')
    print("DON Simulation Done")

    import ex3.ex3_1.visualization as visualization
    fem_results_fig_name = "ex3_1//results//fem_results.png"
    visualization.plot_2d_contour(node=fem_node, results=fem_T,
                                  x0=don_domain_hole_x_center, y0=don_domain_hole_y_center, r=radius,
                                  figname=fem_results_fig_name, is_show=False)

    T_oneBC = don_T[don_hole_outline_indices]
    don_visual_points, don_visual_T = visualization.get_don_domain_results(net=don_element.DON_info.net,
                                                                           x_len=don_domain_hole_width,
                                                                           y_len=don_domain_hole_height,
                                                                           T_BC=T_oneBC, r=radius,
                                                                           don_visual_node_num=100, x0=0.4, y0=0.4,
                                                                           is_zeromean=is_branch_zeromean)
    don_visual_points = np.append(don_node, don_visual_points, axis=0)
    don_visual_T = np.append(don_T, don_visual_T)
    don_results_fig_name = "ex3_1//results//don_results.png"
    visualization.plot_2d_contour(node=don_visual_points, results=don_visual_T,
                                  x0=don_domain_hole_x_center, y0=don_domain_hole_y_center, r=radius,
                                  figname=don_results_fig_name, is_show=False)

    # Visualization of the difference
    diff_results_fig_name = "ex3_1//results//diff_results.png"
    visualization.plot_2d_diff_contourv2(node1=fem_node, results1=fem_T, node2=don_visual_points, results2=don_visual_T,
                                         x0=don_domain_hole_x_center, y0=don_domain_hole_y_center, r=radius,
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
    plt.savefig("ex3_1//results//combined results.png")
    plt.show()


if __name__ == '__main__':
    run_bc1()
    run_bc2()