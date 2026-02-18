from linear_permeability import main as linear_permeability
linear_permeability.run(net_name=".//deeponet_linear_darcy.pth")

from nonlinear_permeability import main as ex4_4_main
# data_path = "nonlinear_permeability//training_data//"
# data_set_name = "darcy_flow.4"
# ex4_4_gendata.ex4_4_data_gen()
ex4_4_main.run("deeponet_nonlinear_darcy.pth")