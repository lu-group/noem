from ex2.solver.fem_solver import solver
from src.util.grf import grf_1Dv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from util.grf import batch_grf_1d


def collect_data(node_coords, mesh, U_range, ele_num, num_samples, k_list, c_list, ele_len=1/50, is_skip=False):
    ele_loc = [0.5 * (node_coords[i] + node_coords[i + 1]) for i in range(ele_num)]
    trunk_input = node_coords
    label = []
    desc = "Collecting data"
    qbar = tqdm(range(num_samples), desc=desc)
    bc_branch_input = []
    k_branch_input = batch_grf_1d(np.array(ele_loc), l=0.3, batch_size=num_samples, sigma=1.0, jitter=1e-8)
    k_branch_input = np.exp(k_branch_input)
    for i in qbar:
        u1 = np.random.uniform(U_range[0], U_range[1])
        u2 = np.random.uniform(U_range[0], U_range[1])
        if 500 <= i and i <= 700:
            u1 = 0. + 0.001 * u1
            u2 = 0.1 + u2 / 5
        U_BC = [u1, u2]
        bc_branch_input.append(U_BC)
        U_BC_nodeid = [0, ele_num]
        k_list = k_branch_input[i]
        U = solver(node=node_coords, mesh=mesh, U_BC=U_BC, U_nodeid=U_BC_nodeid, f_BC=[], f_nodeid=[], k_list=k_list, c_list=c_list * 0)
        if is_skip:
            U = U[::L]
        label += list(U)
    if is_skip:
        return bc_branch_input, k_branch_input, trunk_input[::L], label
    else:
        return bc_branch_input, k_branch_input, trunk_input, label

def save_results(bc_branch_input, k_branch_input, trunk_input, label, path, filename="ex4.1"):
    # Check if the path exists
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, filename)
    columns = ["U" + str(i) for i in range(len(bc_branch_input[0]))]
    bc_branch_input_df = pd.DataFrame(bc_branch_input, columns=columns)
    bc_branch_input_df.to_csv(filename + "_bc_branch_input", index=False)

    columns = ["k" + str(i) for i in range(len(k_branch_input[0]))]
    f_branch_input_df = pd.DataFrame(k_branch_input, columns=columns)
    f_branch_input_df.to_csv(filename + "_k_branch_input", index=False)

    trunk_input_df = pd.DataFrame(trunk_input, columns=["x"])
    trunk_input_df.to_csv(filename + "_trunk_input", index=False)

    label_df = pd.DataFrame(label, columns=["U"])
    label_df.to_csv(filename + "_label", index=False)
    print("Data saved!")
    return


if __name__ == '__main__':
    for L in [1]:
        print("Generating data for L =", L)
        ele_num = 50 * L
        ele_len = 1 / 50
        node = np.linspace(0, L, ele_num + 1)
        mesh = [[i, i + 1] for i in range(ele_num)]
        U_range = [-0.5, 0.5]

        k_list = np.ones(ele_num)
        c_list = np.ones(ele_num)

        num_samples = 200
        bc_branch_input, k_branch_input, trunk_input, label = collect_data(node_coords=node, mesh=mesh, U_range=U_range,
                                                                           ele_num=ele_num, k_list=k_list, c_list=c_list,
                                                                           num_samples=num_samples, ele_len=ele_len, is_skip=True)

        save_results(bc_branch_input=bc_branch_input, k_branch_input=k_branch_input, trunk_input=trunk_input, label=label,
                     path="training_data/", filename="L=" + str(L) + "_test_set")

        num_samples = 100
        bc_branch_input, k_branch_input, trunk_input, label = collect_data(node_coords=node, mesh=mesh, U_range=U_range, ele_num=ele_num,
                                                        k_list=k_list, c_list=c_list,  num_samples=num_samples, ele_len=ele_len, is_skip=False)
        save_results(bc_branch_input=bc_branch_input, k_branch_input=k_branch_input, trunk_input=trunk_input, label=label,
                     path="training_data/", filename="L=" + str(L) + "_val_set")

        num_samples = 5000
        bc_branch_input, k_branch_input, trunk_input, label = collect_data(node_coords=node, mesh=mesh, U_range=U_range,
                                                                           ele_num=ele_num, k_list=k_list, c_list=c_list,
                                                                           num_samples=num_samples, ele_len=ele_len, is_skip=True)
        save_results(bc_branch_input=bc_branch_input, k_branch_input=k_branch_input, trunk_input=trunk_input, label=label,
                     path="training_data/", filename="L=" + str(L) + "_train_set")
