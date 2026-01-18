# Training process for Multiple-input DeepONet with the fixed branch input for each epoch
import os
import torch.nn as nn
from src.util.mideeponet import MIDeepONet
from data_driven_training.train_mideeponet_br_tr_fixed import train

def train_model(data_path, sample_name, model_path, model_name):
    tra_perfix = sample_name + "_train_"
    test_perfix = sample_name + "_test_"
    branch_input1_name = "K_branch_input"  # "ux_branch_input"
    branch_input2_name = "u_branch_input"  # "uy_branch_input"
    trunk_name = "trunk_input"
    label_name = "label"  # "ux_label"
    training_sample_name = {"branchinput1": tra_perfix + branch_input1_name,
                            "branchinput2": tra_perfix + branch_input2_name,
                            "trunk": tra_perfix + trunk_name,
                            "label": tra_perfix + label_name}
    test_sample_name = {"branchinput1": test_perfix + branch_input1_name,
                        "branchinput2": test_perfix + branch_input2_name,
                        "trunk": test_perfix + trunk_name,
                        "label": test_perfix + label_name}

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    batch_size = 300
    TOL = 1e-9

    output_num = 150

    trunk_hidden_layer_num = 4
    trunk_neuron_num = 150
    trunk_hidden_sizes = [trunk_neuron_num] * trunk_hidden_layer_num
    trunk_act_fn = [nn.Tanh()] * trunk_hidden_layer_num

    branch_hidden_layer_num = 4
    branch_neuron_num = 150
    branch_hidden_sizes = [branch_neuron_num] * branch_hidden_layer_num
    branch_act_fn1 = [nn.Tanh()] * branch_hidden_layer_num
    branch_act_fn2 = [nn.Tanh()] * branch_hidden_layer_num

    branchinfo1 = {'act_fn': branch_act_fn1, 'input_size': 400, 'output_size': output_num,
                   'hidden_sizes': branch_hidden_sizes}
    branchinfo2 = {'act_fn': branch_act_fn2, 'input_size': 80, 'output_size': output_num,
                   'hidden_sizes': branch_hidden_sizes}
    trunkinfo = {'act_fn': trunk_act_fn, 'input_size': 2, 'output_size': output_num,
                 'hidden_sizes': trunk_hidden_sizes}

    channel_size = [output_num]
    mideeponet = MIDeepONet(branchinfo_list=[branchinfo1, branchinfo2], trunkinfo=trunkinfo, channel_size=channel_size)

    import src.util.cnn_darcy as cnn
    import torch.nn.functional as F
    cnn = cnn.CNN(activation1=F.relu, activation2=F.tanh, outputsize=output_num)
    mideeponet.branchnet_list[0] = cnn

    lr = 0.0002

    max_epoch = 100000
    sample_num = None
    train_trunk_sample_num = None
    test_trunk_sample_num = None
    train(net=mideeponet, batch_size=batch_size, max_epoch=max_epoch, TOL=TOL,
          data_path=data_path, training_sample_name=training_sample_name, test_sample_name=test_sample_name,
          model_path=model_path, model_name=model_name, sample_num=sample_num,
          train_trunk_sample_num=train_trunk_sample_num, test_trunk_sample_num=test_trunk_sample_num,
          device=None, lr=0.001, early_stopping_epoch=5000, is_loss_plot=True)