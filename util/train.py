import numpy as np
import os
from tqdm import *
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from util.deeponet import DeepONet

class LossEvaluation:
    loss_training = [] # Each element is a list of loss terms
    loss_test = [] # A list of loss terms in test set
    fig, axs = None, None # Parameters for plotting

    @staticmethod
    def update_loss_training(loss_terms):
        for i in range(len(loss_terms)):
            if len(LossEvaluation.loss_training) <= i:
                LossEvaluation.loss_training.append([])
            LossEvaluation.loss_training[i].append(loss_terms[i])

    @staticmethod
    def update_loss_test(loss_terms):
        LossEvaluation.loss_test.append(loss_terms)

    @staticmethod
    def update_loss_test(loss_terms):
        LossEvaluation.loss_test.append(loss_terms)

    @staticmethod
    def init_plot(max_epoch):
        LossEvaluation.max_epoch = max_epoch
        LossEvaluation.fig, LossEvaluation.axs = plt.subplots(1, 2, figsize=(12, 5))
        LossEvaluation.fig.show()
        LossEvaluation.fig.canvas.draw()
        plt.pause(0.05)

    @staticmethod
    def update_plot():
        if LossEvaluation.fig is None or LossEvaluation.axs is None:
            LossEvaluation.init_plot()

        LossEvaluation.axs[0].clear()
        LossEvaluation.axs[1].clear()

        for i, loss_term in enumerate(LossEvaluation.loss_training):
            LossEvaluation.axs[0].plot(loss_term, label=f'Train Loss Term {i + 1}')
        LossEvaluation.axs[0].set_title('Training Loss Terms')
        LossEvaluation.axs[0].set_xlabel('Epoch')
        LossEvaluation.axs[0].set_xlim(0, LossEvaluation.max_epoch)
        LossEvaluation.axs[0].set_ylabel('Loss')
        LossEvaluation.axs[0].set_yscale('log')
        LossEvaluation.axs[0].legend()

        LossEvaluation.axs[1].plot(LossEvaluation.loss_test, label='Test Loss', color='red')
        LossEvaluation.axs[1].set_title('Test Loss')
        LossEvaluation.axs[1].set_xlabel('Epoch')
        LossEvaluation.axs[1].set_xlim(0, LossEvaluation.max_epoch)
        LossEvaluation.axs[1].set_ylabel('Loss')
        LossEvaluation.axs[1].set_yscale('log')
        LossEvaluation.axs[1].legend()

        LossEvaluation.fig.canvas.draw()
        plt.pause(0.05)

class creatDeepONetDataset(Dataset):
    def __init__(self, data_path, name, sample_num, device):

class creatDataSet(Dataset):
    def __init__(self, data_path, name, sample_num, device, input_scaler=None, output_scaler=None):
        input_fileName = data_path + name + "_input"
        #  read the first 1000 samples
        input_data = pd.read_csv(input_fileName, sep=",", nrows=sample_num)
        tinput_tensor = torch.tensor(np.array(input_data, dtype=np.float32))
        if input_scaler is None:
            self.input_tensor, self.input_scaler = self.transfer_input(tinput_tensor)
        else:
            self.input_tensor = self.transfer_input(tinput_tensor, input_scaler=input_scaler)
            self.input_scaler = input_scaler
        self.input_tensor = self.input_tensor.to(device)
        label_fileName = data_path + name + "_label"
        label_data = pd.read_csv(label_fileName, sep=",", nrows=sample_num)
        tlabel_tensor = torch.tensor(np.array(label_data, dtype=np.float32))
        if output_scaler is None:
            self.label_tensor, self.output_scaler = self.transfer_label(tlabel_tensor)
        else:
            self.label_tensor = self.transfer_label(tlabel_tensor, output_scaler=output_scaler)
            self.output_scaler = output_scaler
        self.label_tensor = self.label_tensor.to(device)

    def __getitem__(self, idx):
        return self.input_tensor[idx], self.label_tensor[idx]

    def __len__(self):
        return self.label_tensor.size(0)

    def transfer_input(self, input_tensor, input_scaler=None):
        if input_scaler is None:
            input_scaler = []
            for ii in range(input_tensor.size(1)):
                tscaler = float(torch.max(torch.abs(input_tensor[:, ii])))
                input_scaler.append(tscaler)
                input_tensor[:, ii] = input_tensor[:, ii] / tscaler
            return input_tensor, input_scaler
        else:
            for ii in range(input_tensor.size(1)):
                input_tensor[:, ii] = input_tensor[:, ii] / input_scaler[ii]
            return input_tensor

    def transfer_label(self, label_tensor, output_scaler=None):
        if output_scaler is None:
            output_scaler = []
            for ii in range(label_tensor.size(1)):
                output_scaler.append(float(torch.max(torch.abs(label_tensor[:, ii]))))
                label_tensor[:, ii] = label_tensor[:, ii] / output_scaler[ii]
            return label_tensor, output_scaler
        else:
            for ii in range(label_tensor.size(1)):
                label_tensor[:, ii] = label_tensor[:, ii] / output_scaler[ii]
            return label_tensor

def train(net, batch_size, max_epoch, TOL, data_path, sample_name, model_path, model_name, sample_num, device=None, lr=0.001, weight_decay=0.0001, is_loss_plot=False):
    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("The device is set to be %s." % device)
    train_dataset = creatDataSet(data_path, (sample_name + "_train"), sample_num=sample_num, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    input_scaler, output_scaler = train_dataset.input_scaler, train_dataset.output_scaler
    net.config["input_scaler"] = input_scaler; net.config["output_scaler"] = output_scaler
    test_dataset = creatDataSet(data_path, (sample_name + "_test"), sample_num=None, input_scaler=input_scaler, output_scaler=output_scaler, device=device)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    loss_MIN = 1e10
    # Lists to store the loss values
    train_loss_history = []
    test_loss_history = []

    desc = "start training..."
    pbar = tqdm(range(max_epoch), desc=desc)
    iterCounter = 0
    if is_loss_plot:
        LossEvaluation.init_plot(max_epoch)
    for current_epoch in pbar:
        net.train()
        tloss_value = []
        loss_list = []
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            predict_y = net(train_x).to(device)
            opt.zero_grad()
            loss = loss_fn(predict_y, train_label)
            loss_list.append(loss.item())
            tloss_value.append(loss.item())
            loss.backward()
            opt.step()
        train_loss_history.append(np.mean(tloss_value))
        if is_loss_plot:
            LossEvaluation.update_loss_training([np.mean(loss_list)])
        net.eval()

        tloss_value = []
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = net(test_x.float()).to(device).detach()
            loss = loss_fn(predict_y, test_label)
            tloss_value.append(loss.item())
            LossEvaluation.update_loss_test(loss.item())
        test_loss_history.append(np.mean(tloss_value))

        pbar.set_description("Epoch %d, Train Loss %.2e, Test Loss %.2e, Minimum Loss %.2e" % (current_epoch, train_loss_history[-1], test_loss_history[-1], loss_MIN))

        if current_epoch % 20 == 0 and is_loss_plot:
            LossEvaluation.update_plot()

        iterCounter += 1
        if loss.item() < loss_MIN:
            iterCounter = 0
            loss_MIN = loss.item()
            try:
                torch.save(net, model_path + model_name + ".pt")
            except:
                pass

        if iterCounter > 100000:
            print("The loss value is not decreasing. Stop training.")
            return

        if loss.item() < TOL:
            print("Tolerance is satisfied! Stop training.")
            torch.save(net, model_path + model_name + ".pt")
            break


    print("The current loss value is %.2e. Stop training."% loss_MIN)
    # NetworkSL.save_model(model, model_path, name)
    return

if __name__ == '__main__':
    data_path = os.getcwd() + r"\\"
    model_path = os.getcwd() + r"\\"
    sample_name = "multiscale_1d_problem.1"
    model_name = "multiscale_1d_problem.1_donv2"
    batch_size = 1000
    TOL = 1e-8
    # branchinfo = {'act_fn': [nn.Tanh(), nn.Tanh(), nn.Tanh()], 'input_size': 64, 'output_size': 20,
    #               'hidden_sizes': [128,128,128]}
    # trunkinfo = {'act_fn': [nn.Tanh(), nn.Tanh(), nn.Tanh()], 'input_size': 2, 'output_size': 20,
    #              'hidden_sizes': [64,64,64]}
    # channel_size = [20]
    # deeponet = DeepONet(branchinfo, trunkinfo, channel_size)
    deeponet = torch.load(model_path + model_name + ".pt")
    lr = 0.001
    max_epoch = 8000
    sample_num = None
    train(deeponet, batch_size, max_epoch, TOL, data_path, sample_name, model_path, model_name, sample_num=sample_num,
          device=None, lr=lr, weight_decay=0)
