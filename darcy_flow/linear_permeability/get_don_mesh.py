import numpy as np

def get_single_don_mesh(x0, y0, x_len, y_len, x_mesh_num, y_mesh_num):
    buttom_nodes = [[x0 + x_len / x_mesh_num * i, y0] for i in range(x_mesh_num + 1)]
    right_nodes = [[x0 + x_len, y0 + y_len / y_mesh_num * i] for i in range(y_mesh_num + 1)]
    top_nodes = [[x0 + x_len - x_len / x_mesh_num * i, y0 + y_len] for i in range(x_mesh_num + 1)]
    left_nodes = [[x0, y0 + y_len - y_len / y_mesh_num * i] for i in range(y_mesh_num + 1)]
    don_mesh = buttom_nodes[:-1] + right_nodes[:-1] + top_nodes[:-1] + left_nodes[:-1]
    don_mesh = np.array(don_mesh)
    return don_mesh

def get_don_mesh(x_len, y_len, x_grid_num, y_grid_num, x_mesh_num, y_mesh_num):
    # Define the geometry of the domain
    don_mesh_list = []
    for i in range(x_grid_num):
        for j in range(y_grid_num):
            x0 = x_len / x_grid_num * i
            y0 = y_len / y_grid_num * j
            don_mesh = get_single_don_mesh(x0, y0, x_len / x_grid_num, y_len / y_grid_num, x_mesh_num, y_mesh_num)
            don_mesh_list.append(don_mesh)
    flatten_don_mesh = np.concatenate(don_mesh_list)
    # Create a node list removing all the repeated nodes
    node_list = []
    for node in flatten_don_mesh:
        if list(node) not in node_list:
            node_list.append(list(node))
    don_mesh_idx_list = []
    for don_mesh in don_mesh_list:
        don_mesh_idx = []
        for node in don_mesh:
            idx = node_list.index(list(node))
            don_mesh_idx.append(idx)
    return node_list, don_mesh_idx_list