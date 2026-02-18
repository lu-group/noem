from main import *
from tqdm import *
def get_noe_acc(net_name, test_num, U_range=[-0.1,0.1]):
    error_list = []
    qbar = tqdm(range(test_num))
    for i in qbar:
        U1 = np.random.uniform(U_range[0], U_range[1])
        U2 = np.random.uniform(U_range[0], U_range[1])
        x_fem, U_fem = single_fem_run(U1, U2)
        don_segment = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
        don_segment = np.array(don_segment) / 8
        don_segment = don_segment.tolist()
        x_don, U_don, noe_x_fem, noe_U_fem, noe_x_don, noe_U_don = single_noe_runv2hc(don_segment, U1, U2, net_name)
        relative_L2_error = np.linalg.norm(U_don - U_fem) / np.linalg.norm(U_fem)
        error_list.append(relative_L2_error)
    return np.mean(error_list)

