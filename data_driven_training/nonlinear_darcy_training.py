# Training process for Multiple-input DeepONet with the fixed branch input for each epoch
import os
import torch.nn as nn
from src.util.deeponet import DeepONet
from data_driven_training.train_deeponet_br_tr_fixedv2 import train

def train_model(data_path, sample_name, model_path, model_name):
    import os
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    tra_perfix = sample_name + "_train_"
    test_perfix = sample_name + "_test_"
    branch_input_name = "branch_input"
    trunk_name = "trunk_input"
    label_name = "label"
    training_sample_name = {"branchinput": tra_perfix + branch_input_name,
                            "trunk": tra_perfix + trunk_name,
                            "label": tra_perfix + label_name}
    test_sample_name = {"branchinput": test_perfix + branch_input_name,
                        "trunk": test_perfix + trunk_name,
                        "label": test_perfix + label_name}
    batch_size = 500
    TOL = 1e-9
    output_num = 128
    trunk_hidden_layer_num = 4
    trunk_neuron_num = output_num
    trunk_hidden_sizes = [trunk_neuron_num] * trunk_hidden_layer_num
    trunk_act_fn = [nn.Tanh()] * trunk_hidden_layer_num

    branch_hidden_layer_num = 4
    branch_neuron_num = output_num
    branch_hidden_sizes = [branch_neuron_num] * branch_hidden_layer_num
    branch_act_fn = [nn.Tanh()] * branch_hidden_layer_num

    branchinfo = {'act_fn': branch_act_fn, 'input_size': 80, 'output_size': output_num,
                  'hidden_sizes': branch_hidden_sizes}
    trunkinfo = {'act_fn': trunk_act_fn, 'input_size': 2, 'output_size': output_num,
                 'hidden_sizes': trunk_hidden_sizes}

    channel_size = [output_num]
    deeponet = DeepONet(branchinfo=branchinfo, trunkinfo=trunkinfo, channel_size=channel_size)

    max_epoch = 100000
    sample_num = None
    train_trunk_sample_num = None
    test_trunk_sample_num = None
    train(net=deeponet, batch_size=batch_size, max_epoch=max_epoch, TOL=TOL,
          data_path=data_path, training_sample_name=training_sample_name, test_sample_name=test_sample_name,
          model_path=model_path, model_name=model_name, sample_num=sample_num,
          train_trunk_sample_num=train_trunk_sample_num, test_trunk_sample_num=test_trunk_sample_num,
          device=None, lr=0.001, early_stopping_epoch=5000, is_loss_plot=True, loss_type="L2")