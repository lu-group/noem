import gmsh
import numpy as np
import matplotlib.pyplot as plt

def get_mesh(mesh_size, width, height, center_x_list, center_y_list, radius_list, mesh_size_circle=None, mesh_logging=False):
    gmsh.initialize()
    if not mesh_logging:
        gmsh.option.setNumber("General.Verbosity", 0)
    # Use a smaller mesh size for the circle if provided, otherwise use half of the main mesh size
    if mesh_size_circle is None:
        mesh_size_circle = mesh_size / 2

    # Start a new model
    gmsh.model.add("rectangle_with_hole")

    # Add rectangle points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(width, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(width, height, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)

    # Connect the points to create the outer boundary (rectangle)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Create a loop for the outer boundary
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    circle_loops = []
    # Generate each circle (hole)
    for center_x, center_y, radius in zip(center_x_list, center_y_list, radius_list):
        pc = gmsh.model.geo.addPoint(center_x, center_y, 0, mesh_size_circle)
        p5 = gmsh.model.geo.addPoint(center_x, center_y - radius, 0, mesh_size_circle)
        p6 = gmsh.model.geo.addPoint(center_x, center_y + radius, 0, mesh_size_circle)
        arc1 = gmsh.model.geo.addCircleArc(p5, pc, p6)
        arc2 = gmsh.model.geo.addCircleArc(p6, pc, p5)
        circle_loop = gmsh.model.geo.addCurveLoop([arc1, arc2])
        circle_loops.append(circle_loop)

    # Create a plane surface with all holes
    gmsh.model.geo.addPlaneSurface([outer_loop] + circle_loops)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Fetch node and element data
    node_ids, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords).reshape(-1, 3)[:,
                  :2]  # Reshape to a 2D array for X, Y, Z coordinates and select X, Y

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

    # Identify the outline nodes of the rectangle
    outline_indices = []
    outline_nodes = []
    for idx, node in enumerate(filtered_node_coords):
        if (node[0] == 0 or node[0] == width) and (0 <= node[1] <= height):
            outline_indices.append(used_node_indices[idx])
            outline_nodes.append(node)
        elif (node[1] == 0 or node[1] == height) and (0 <= node[0] <= width):
            outline_indices.append(used_node_indices[idx])
            outline_nodes.append(node)

    outline_nodes = np.array(outline_nodes)
    outline_indices = np.array(outline_indices)

    # Remove duplicates by keeping unique rows
    unique_outline_indices, unique_idx = np.unique(outline_indices, return_index=True)
    unique_outline_nodes = outline_nodes[unique_idx]

    # Sort the unique outline nodes counter-clockwise starting from the lower left corner
    sorted_outline_nodes = unique_outline_nodes[np.lexsort((unique_outline_nodes[:, 1], unique_outline_nodes[:, 0]))]

    bottom_nodes = sorted_outline_nodes[sorted_outline_nodes[:, 1] == 0]
    right_nodes = sorted_outline_nodes[sorted_outline_nodes[:, 0] == width]
    top_nodes = sorted_outline_nodes[sorted_outline_nodes[:, 1] == height][::-1]
    left_nodes = sorted_outline_nodes[sorted_outline_nodes[:, 0] == 0][::-1]

    bottom_nodes = bottom_nodes[:-1]
    right_nodes = right_nodes[:-1]
    top_nodes = top_nodes[:-1]
    left_nodes = left_nodes[:-1]

    sorted_outline_nodes = np.vstack([bottom_nodes, right_nodes, top_nodes, left_nodes])

    sorted_outline_indices = []
    for _node in sorted_outline_nodes:
        # Obtain the index of the node
        idx = np.where((filtered_node_coords == _node).all(axis=1))[0][0]
        sorted_outline_indices.append(idx)


    sorted_outline_indices = np.array(sorted_outline_indices)
    filtered_element_info = np.array(filtered_element_info) - 1
    # Finalize the Gmsh API
    gmsh.finalize()
    return filtered_node_coords, filtered_element_info, sorted_outline_indices


if __name__ == "__main__":
    width = 0.8
    height = 0.8
    mesh_size = 0.1
    radius_list = [0.05, 0.05]
    mesh_size_circle = 0.01
    center_x_list = [0.3, 0.5]
    center_y_list = [0.3, 0.5]
    node, element, outline_indices = get_mesh(mesh_size=mesh_size, width=width, height=height,
                                              center_x_list=center_x_list, center_y_list=center_y_list,radius_list=radius_list,
                                          mesh_size_circle=mesh_size_circle)

    # Print outline node indices and their locations
    print("\nOutline Node Indices (Counter-clockwise):\n")
    for i in range(len(outline_indices)):
        print(f"Outline Node {i + 1}: {outline_indices[i]}, Location: {node[int(outline_indices[i])]}")

    # Plot the mesh
    plt.figure(figsize=(8, 6))
    for tri in element:
        triangle = plt.Polygon(node[tri, :2], edgecolor='black', facecolor='none')
        plt.gca().add_patch(triangle)

    plt.scatter(node[:, 0], node[:, 1], c='red', marker='o')  # Node points
    plt.plot(node[outline_indices, 0], node[outline_indices, 1], 'bo-', label='Outline Nodes')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Mesh Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()
