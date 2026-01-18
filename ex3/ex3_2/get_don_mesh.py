import gmsh
import numpy as np
import matplotlib.pyplot as plt

def generate_mesh(mesh_size, outer_width, outer_height, hole_width_list, hole_height_list,
                  hole_x_center_list, hole_y_center_list, mesh_size_hole=None, mesh_logging=False):
    gmsh.initialize()
    if not mesh_logging:
        gmsh.option.setNumber("General.Verbosity", 0)
    # Start a new model
    gmsh.model.add("rectangle_with_hole")

    # Add outer rectangle points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(outer_width, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(outer_width, outer_height, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, outer_height, 0, mesh_size)

    # Connect the points to create the outer boundary (rectangle)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Create a loop for the outer boundary
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    hole_loop_list = []
    # Generate each circle (hole)
    for hole_x_center, hole_y_center, hole_width, hole_height in zip(hole_x_center_list, hole_y_center_list, hole_width_list, hole_height_list):
        p5 = gmsh.model.geo.addPoint(hole_x_center - hole_width / 2, hole_y_center - hole_height / 2, 0, mesh_size_hole)
        p6 = gmsh.model.geo.addPoint(hole_x_center + hole_width / 2, hole_y_center - hole_height / 2, 0, mesh_size_hole)
        p7 = gmsh.model.geo.addPoint(hole_x_center + hole_width / 2, hole_y_center + hole_height / 2, 0, mesh_size_hole)
        p8 = gmsh.model.geo.addPoint(hole_x_center - hole_width / 2, hole_y_center + hole_height / 2, 0, mesh_size_hole)

        l5 = gmsh.model.geo.addLine(p5, p6)
        l6 = gmsh.model.geo.addLine(p6, p7)
        l7 = gmsh.model.geo.addLine(p7, p8)
        l8 = gmsh.model.geo.addLine(p8, p5)

        hole_loop = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
        hole_loop_list.append(hole_loop)

    # Create a plane surface with all holes
    gmsh.model.geo.addPlaneSurface([outer_loop] + hole_loop_list)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Fetch node and element data
    node_ids, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]  # Reshape to a 2D array for X, Y, Z coordinates and select X, Y

    element_types, element_ids, element_tags = gmsh.model.mesh.getElements()
    element_info = []

    for etype, nodes in zip(element_types, element_tags):
        nodes_per_element = gmsh.model.mesh.getElementProperties(etype)[3]
        if nodes_per_element == 3:  # Ensure we only consider triangular elements
            element_nodes = np.array(nodes).reshape(-1, nodes_per_element)
            element_info.extend(element_nodes)

    element_info = np.array(element_info)

    # Create a set of used node indices
    used_node_indices = np.unique(element_info.flatten())

    # Filter out unused nodes
    filtered_node_coords = node_coords[used_node_indices - 1]
    index_mapping = {old_index: new_index for new_index, old_index in enumerate(used_node_indices, start=1)}
    filtered_element_info = np.vectorize(index_mapping.get)(element_info)

    def sort_nodes_counter_clockwise(nodes):
        nodes = np.array(nodes)
        x_min = np.min(nodes[:, 0])
        x_max = np.max(nodes[:, 0])
        y_min = np.min(nodes[:, 1])
        y_max = np.max(nodes[:, 1])
        sorted_nodes = nodes[np.lexsort((nodes[:, 1], nodes[:, 0]))]

        bottom_nodes = sorted_nodes[sorted_nodes[:, 1] == y_min]
        right_nodes = sorted_nodes[sorted_nodes[:, 0] == y_max]
        top_nodes = sorted_nodes[sorted_nodes[:, 1] == x_max][::-1]
        left_nodes = sorted_nodes[sorted_nodes[:, 0] == x_min][::-1]

        bottom_nodes = bottom_nodes[:-1]
        right_nodes = right_nodes[:-1]
        top_nodes = top_nodes[:-1]
        left_nodes = left_nodes[:-1]

        return np.vstack([bottom_nodes, right_nodes, top_nodes, left_nodes])

    # Identify the outline nodes of the outer rectangle
    outer_outline_indices = []
    outer_outline_nodes = []
    for idx, node in enumerate(filtered_node_coords):
        if (node[0] == 0 or node[0] == outer_width) and (0 <= node[1] <= outer_height):
            outer_outline_indices.append(used_node_indices[idx])
            outer_outline_nodes.append(node)
        elif (node[1] == 0 or node[1] == outer_height) and (0 <= node[0] <= outer_width):
            outer_outline_indices.append(used_node_indices[idx])
            outer_outline_nodes.append(node)

    outer_outline_nodes = np.array(outer_outline_nodes)
    outer_outline_indices = np.array(outer_outline_indices)

    # Remove duplicates by keeping unique rows
    unique_outer_outline_indices, unique_outer_idx = np.unique(outer_outline_indices, return_index=True)
    unique_outer_outline_nodes = outer_outline_nodes[unique_outer_idx]

    # Sort the unique outline nodes counter-clockwise starting from the lower left corner
    sorted_outer_outline_nodes = sort_nodes_counter_clockwise(unique_outer_outline_nodes)
    sorted_outer_outline_indices = []
    for _node in sorted_outer_outline_nodes:
        # Obtain the index of the node
        idx = np.where((filtered_node_coords == _node).all(axis=1))[0][0]
        sorted_outer_outline_indices.append(idx)
    sorted_outer_outline_indices = np.array(sorted_outer_outline_indices)

    # Finalize the Gmsh API
    gmsh.finalize()
    filtered_element_info = np.array(filtered_element_info, dtype=int) - 1
    return filtered_node_coords, filtered_element_info, sorted_outer_outline_indices

def get_hole_outline_indices_list(outer_width, outer_height, hole_width_list, hole_height_list, hole_x_center_list,
                                  hole_y_center_list, node, element, outer_outline_indices):
    sorted_hole_outline_indices_list = []
    for hole_x_center, hole_y_center, hole_width, hole_height in zip(hole_x_center_list, hole_y_center_list,
                                                                     hole_width_list, hole_height_list):

        # Identify the outline nodes of the hole
        hole_outline_indices = []
        hole_outline_nodes = []
        for idx, tnode in enumerate(node):
            if (tnode[0] == hole_x_center - hole_width / 2 or tnode[0] == hole_x_center + hole_width / 2) and \
                    (hole_y_center - hole_height / 2 <= tnode[1] <= hole_y_center + hole_height / 2):
                hole_outline_indices.append(idx)
                hole_outline_nodes.append(node)
            elif (tnode[1] == hole_y_center - hole_height / 2 or tnode[1] == hole_y_center + hole_height / 2) and \
                    (hole_x_center - hole_width / 2 <= tnode[0] <= hole_x_center + hole_width / 2):
                hole_outline_indices.append(idx)
        # Remove duplicates by keeping unique rows
        unique_hole_outline_indices, unique_hole_idx = np.unique(hole_outline_indices, return_index=True)
        unique_hole_locations = node[unique_hole_outline_indices]
        # Sort the unique outline nodes counter-clockwise starting from the lower left corner
        def sort_nodes_counter_clockwise(unique_hole_locations, unique_hole_outline_indices):
            x_min = np.min(unique_hole_locations[:, 0])
            x_max = np.max(unique_hole_locations[:, 0])
            y_min = np.min(unique_hole_locations[:, 1])
            y_max = np.max(unique_hole_locations[:, 1])
            sorted_hole_locations = unique_hole_locations[np.lexsort((unique_hole_locations[:, 1], unique_hole_locations[:, 0]))]

            bottom_hole_nodes = sorted_hole_locations[sorted_hole_locations[:, 1] == y_min]
            right_hole_nodes = sorted_hole_locations[sorted_hole_locations[:, 0] == x_max]
            top_hole_nodes = sorted_hole_locations[sorted_hole_locations[:, 1] == y_max][::-1]
            left_hole_nodes = sorted_hole_locations[sorted_hole_locations[:, 0] == x_min][::-1]

            bottom_hole_nodes = bottom_hole_nodes[:-1]
            right_hole_nodes = right_hole_nodes[:-1]
            top_hole_nodes = top_hole_nodes[:-1]
            left_hole_nodes = left_hole_nodes[:-1]

            sorted_hole_locations = np.vstack([bottom_hole_nodes, right_hole_nodes, top_hole_nodes, left_hole_nodes])

            sorted_hole_outline_indices = []
            for _node in sorted_hole_locations:
                # Obtain the index of the node
                idx = np.where((unique_hole_locations == _node).all(axis=1))[0][0]
                sorted_hole_outline_indices.append(unique_hole_outline_indices[idx])

            return np.array(sorted_hole_outline_indices)
        sorted_hole_outline_indices = sort_nodes_counter_clockwise(unique_hole_locations, unique_hole_outline_indices)
        sorted_hole_outline_indices_list.append(sorted_hole_outline_indices)
    return sorted_hole_outline_indices_list

def generate_don_mesh(mesh_size, outer_width, outer_height, hole_width_list, hole_height_list, hole_x_center_list,
                      hole_y_center_list, mesh_size_hole):
    node, element, outer_outline_indices = generate_mesh(mesh_size, outer_width, outer_height,
                                                         hole_width_list, hole_height_list, hole_x_center_list,
                                                         hole_y_center_list, mesh_size_hole)
    hole_outline_indices_list = get_hole_outline_indices_list(outer_width, outer_height,
                                                              hole_width_list, hole_height_list,
                                                              hole_x_center_list, hole_y_center_list,
                                                              node, element, outer_outline_indices)
    return node, element, outer_outline_indices, hole_outline_indices_list
if __name__ == "__main__":
    outer_width = 4.5
    outer_height = outer_width
    mesh_size = 0.1
    hole_width_list = []
    hole_height_list = []
    hole_x_center_list = []
    hole_y_center_list = []

    for i in range(3):
        tx_center = 0.75 + i * 1.5
        for j in range(3):
            ty_center = 0.75 + j * 1.5
            hole_width_list.append(0.8)
            hole_x_center_list.append(tx_center)
            hole_height_list.append(0.8)
            hole_y_center_list.append(ty_center)
    mesh_size_hole = 0.05

    node, element, outer_outline_indices, hole_outline_indices_list = generate_don_mesh(mesh_size, outer_width, outer_height,
                                                                               hole_width_list, hole_height_list, hole_x_center_list,
                                                                               hole_y_center_list, mesh_size_hole)

    # Print outline node indices and their locations
    print("\nOuter Outline Node Indices (Counter-clockwise):\n")
    for i in range(len(outer_outline_indices)):
        print(
            f"Outer Outline Node {i + 1}: {outer_outline_indices[i]}, Location: {node[int(outer_outline_indices[i])]}")


    # Plot the mesh
    plt.figure(figsize=(8, 6))
    for tri in element:
        triangle = plt.Polygon(node[tri, :2], edgecolor='black', facecolor='none')
        plt.gca().add_patch(triangle)

    # plt.scatter(node[:, 0], node[:, 1], c='red', marker='o')  # Node points
    # plt.plot(node[outer_outline_indices, 0], node[outer_outline_indices, 1], 'bo-', label='Outer Outline Nodes')

    for hole_outline_indices in hole_outline_indices_list:
        plt.plot(node[hole_outline_indices, 0], node[hole_outline_indices, 1], 'go-', label='Hole Outline Nodes', markersize=0.5)

    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Mesh Visualization')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.grid(True)
    # plt.legend()
    plt.show()
