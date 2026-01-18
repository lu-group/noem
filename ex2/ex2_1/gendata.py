from ex2.solver.fem_solver import solver
from src.util.grf import grf_1Dv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def collect_data(node_coords, mesh, U_range, ele_num, num_samples, k_list, c_list, f_list):
    branch_input = []
    ele_loc = [0.5 * (node_coords[i] + node_coords[i + 1]) for i in range(ele_num)]
    trunk_input = node_coords
    label = []
    desc = "Collecting data"
    qbar = tqdm(range(num_samples), desc=desc)
    f_BC = np.arange(0, len(node_coords))
    for i in qbar:
        u1 = np.random.uniform(U_range[0], U_range[1])
        u2 = np.random.uniform(U_range[0], U_range[1])
        U_BC = [u1, u2]
        branch_input.append(U_BC)
        U_BC_nodeid = [0, ele_num]
        U = solver(node=node, mesh=mesh, U_BC=U_BC, U_nodeid=U_BC_nodeid, f_BC=f_list, f_nodeid=f_BC, k_list=k_list, c_list=c_list)
        label += list(U)
    return branch_input, trunk_input, label

def save_results(branch_input, trunk_input, label, path, filename="ex4.1"):
    # Check if the path exists
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, filename)
    columns = ["U" + str(i) for i in range(len(branch_input[0]))]
    branch_input_df = pd.DataFrame(branch_input, columns=columns)
    branch_input_df.to_csv(filename + "_branch_input", index=False)

    trunk_input_df = pd.DataFrame(trunk_input, columns=["x"])
    trunk_input_df.to_csv(filename + "_trunk_input", index=False)

    label_df = pd.DataFrame(label, columns=["U"])
    label_df.to_csv(filename + "_label", index=False)
    print("Data saved!")
    return


if __name__ == '__main__':
    L = 1/8
    ele_num = 200
    node = np.linspace(0, L, ele_num + 1)
    f_list = 0.5 * np.ones_like(node) * L / ele_num
    mesh = [[i, i + 1] for i in range(ele_num)]
    U_range = [-0.1, 0.1]
    ele_loc = [0.5 * (node[i] + node[i + 1]) for i in range(ele_num)]
    wave_num = 16
    k_list = 0.5 * np.sin(2 * np.pi * wave_num * np.array(ele_loc)) + 0.8
    c_list = np.zeros(ele_num)

    corr_length = 0.3
    num_samples = 200
    branch_input,  trunk_input, label = collect_data(node_coords=node, mesh=mesh, U_range=U_range, ele_num=ele_num, k_list=k_list, c_list=c_list, f_list=f_list, num_samples=num_samples)
    save_results(branch_input=branch_input, trunk_input=trunk_input, label=label,
                 path="training_data/", filename="ex21_test")

    num_samples = 10000
    branch_input, trunk_input, label = collect_data(node_coords=node, mesh=mesh, U_range=U_range, ele_num=ele_num, k_list=k_list, c_list=c_list, f_list=f_list, num_samples=num_samples)
    save_results(branch_input=branch_input, trunk_input=trunk_input, label=label,
                 path="training_data/", filename="ex21_train")
