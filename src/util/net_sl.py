import os, torch, json
from src.util.fcnn import BaseNetwork
act_fn_by_name = {
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    'Tanhshrink': torch.nn.Tanhshrink,
    "Softplus": torch.nn.Softplus,
    "Sigmoid": torch.nn.Sigmoid
}
########################################
def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")

def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")
#########################################

def load_model(model_path, model_name, net=None):
    """
    Loads a saved model from disk.

    Inputs:
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
        net - (Optional) If given, the state dict is loaded into this model. Otherwise, a new model is created.
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    assert os.path.isfile(config_file), f"Could not find the config file \"{config_file}\". Are you sure this is the correct path and you have your model config stored here?"
    assert os.path.isfile(model_file), f"Could not find the model file \"{model_file}\". Are you sure this is the correct path and you have your model stored here?"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    # print("--------------------The model config----------------")
    # print(config_dict)
    # print("----------------------------------------------------")
    if net is None:
        act_fn_name = config_dict.pop("act_fn")
        act_fu = []
        for ii in act_fn_name:
            act_fu.append(act_fn_by_name[ii]())
        net = BaseNetwork(act_fn=act_fu, **config_dict)
    net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    net.config = config_dict
    return net

def save_model(model, model_path, model_name):
    """
    Given a model, we save the state_dict and hyperparameters.

    Inputs:
        model - Network object to save parameters from
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    config_dict = model.config
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    with open(config_file, "w") as f:
        json.dump(config_dict, f)
    torch.save(model.state_dict(), model_file)
