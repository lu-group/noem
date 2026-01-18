# External Lib.
import os, math
# Internal Lib.
from src.fileio.io import loadmodel
from src.FEsolver import staticnonlinear
import src.util.net_sl as NetworkSL
import src.util.visual_model as visual_model

def main(model, filename="", is_log=True, is_visual=False):
    femodel = loadmodel(filename)
    if is_visual:
        visual_model.visualize(femodel)
    return staticnonlinear.run(femodel, model, log=is_log)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = current_dir + r"\examples\\"
    filename = file_path + "fem_template_cache"

    import torch
    model_path = r"C:\Users\Weihang Ouyang\PycharmProjects\deeponet_element_method\data_driven_training\ex2\\"
    model_name = "ex2_deeponetv3.pth"
    deeponet = torch.load(model_path + model_name)
    deeponet.to("cpu")

    main(model=deeponet, filename=filename)