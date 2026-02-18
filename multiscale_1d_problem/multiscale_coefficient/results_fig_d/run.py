width_list = [10, 20, 30, 40, 50, 60, 70, 90, 100]
model_acc_list = []
noe_acc_list = []
from get_model_acc import get_model_acc
from get_noe_acc import get_noe_acc
test_num = 100
u = 0.1

#===================================================================
# Please run multiscale_1d_problem/multiscale_coefficient/gendata.py and data_driven_training\training_ex2_1_fig_d.py first
# Models with randomly-drawn hyperparameters will be trained.
# Then, run this code to generate the data in panel(d).
model_num = 1
for i in range(model_num):
    net_name = r"..\\..\\..\\data_driven_training/multiscale_coefficient/hc_ode_random_" + str(i) + ".pth"
    model_acc = get_model_acc(net_name)
    print(str(i + 1) + "th model accuracy: ", model_acc)
    noe_acc = get_noe_acc(net_name, test_num=test_num, U_range=[-u, u])
    model_acc_list.append(model_acc)
    noe_acc_list.append(noe_acc)
    print(str(i + 1) + "th model NOE accuracy: ", noe_acc)

import numpy as np
saved_results = {"model_acc_list": model_acc_list, "noe_acc_list": noe_acc_list}
np.save("scalability_test_results", saved_results)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(model_acc_list, noe_acc_list)
plt.xlabel("Model Acc")
plt.ylabel("NOE Acc")
plt.show()