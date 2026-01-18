# Training process for Multiple-input DeepONet with the fixed branch input for each epoch
import numpy as np
from tqdm import *
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from data_driven_training.loss_visual import LossEvaluation


class creatDataSet(Dataset):
    def __init__(self, data_path, sample_name, sample_num=None, trunk_sample_num=None,
                 branch_input_min=None, branch_input_max=None,
                 trunk_input_min=None, trunk_input_max=None,
                 output_min=None, output_max=None, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        branchinput_name = sample_name["branchinput"]
        trunk_name = sample_name["trunk"]
        label_name = sample_name["label"]

        # Load the data
        branch_input_filename = data_path + branchinput_name
        branch_input_data = pd.read_csv(branch_input_filename, sep=",")
        self.branch_input_tensor = torch.tensor(np.array(branch_input_data, dtype=np.float32))

        trunk_input_filename = data_path + trunk_name
        trunk_input_data = pd.read_csv(trunk_input_filename, sep=",")
        trunk_input_tensor = torch.tensor(np.array(trunk_input_data, dtype=np.float32))

        label_filename = data_path + label_name
        label_data = pd.read_csv(label_filename, sep=",")
        label_tensor = torch.tensor(np.array(label_data, dtype=np.float32))

        if trunk_sample_num is None:
            self.trunk_sample_num = trunk_input_tensor.size(0)
            self.trunk_input_tensor = trunk_input_tensor
            self.label_tensor = label_tensor
        else:
            self.trunk_sample_num = trunk_sample_num
            shuffle_index = torch.randperm(trunk_input_tensor.size(0))
            trunk_index = shuffle_index[:self.trunk_sample_num]
            original_trunk_sample_num = trunk_input_tensor.size(0)
            if self.trunk_sample_num > original_trunk_sample_num:
                raise ValueError("The trunk sample number is larger than the original trunk sample number.")
            self.trunk_input_tensor = trunk_input_tensor[trunk_index, :]
            lable_index_list = []
            for i in range(self.branch_input_tensor.size(0)):
                tindex = (trunk_index.numpy() + i * original_trunk_sample_num).tolist()
                lable_index_list += tindex
            lable_index = torch.tensor(lable_index_list)
            self.label_tensor = label_tensor[lable_index, :]

        if sample_num is not None:
            self.branch_input_tensor = self.branch_input_tensor[:sample_num, :]
            self.label_tensor = self.label_tensor[:sample_num * self.trunk_sample_num, :]
        else:
            self.trunk_input_tensor = self.trunk_input_tensor[: self.trunk_sample_num, :]

        self.branch_input_tensor, self.branch_input_min, self.branch_input_max = self.min_max_scaler(self.branch_input_tensor, branch_input_min, branch_input_max)
        self.trunk_input_tensor, self.trunk_input_min, self.trunk_input_max = self.min_max_scaler(self.trunk_input_tensor, trunk_input_min, trunk_input_max)
        self.label_tensor, self.output_min, self.output_max = self.min_max_scaler(self.label_tensor, output_min, output_max)

        self.branch_input_tensor = self.branch_input_tensor.to(device)
        self.trunk_input_tensor = self.trunk_input_tensor.to(device)
        self.label_tensor = self.label_tensor.to(device)

        # Obtain the L2 norm of the label tensor
        self.label_norm = torch.norm(self.label_tensor.view(-1, self.trunk_sample_num), dim=1, p=2)
    def __getitem__(self, idx):
        # idx is the index of the branch input data
        # return: the branch input tensor; the trunk input tensor; the label tensor
        return self.branch_input_tensor[idx], self.label_tensor[idx * self.trunk_sample_num: (idx + 1) * self.trunk_sample_num], self.label_norm[idx]

    def __len__(self):
        return self.branch_input_tensor.size(0)

    def normalization(self, unormalized_tensor, tensor_scaler=None):
        if tensor_scaler is None:
            tensor_scaler = []
            for ii in range(unormalized_tensor.size(1)):
                tscaler = float(torch.max(torch.abs(unormalized_tensor[:, ii])))
                tensor_scaler.append(tscaler)
                unormalized_tensor[:, ii] = unormalized_tensor[:, ii] / tscaler
            return unormalized_tensor, tensor_scaler
        else:
            for ii in range(unormalized_tensor.size(1)):
                unormalized_tensor[:, ii] = unormalized_tensor[:, ii] / tensor_scaler[ii]
            return unormalized_tensor, tensor_scaler

    def min_max_scaler(self, tensor, min_value, max_value):
        if min_value is None and max_value is None:
            min_value = torch.min(tensor, dim=0)[0].tolist()
            max_value = torch.max(tensor, dim=0)[0].tolist()
        min_value = torch.tensor(min_value, device=tensor.device)
        max_value = torch.tensor(max_value, device=tensor.device)
        tensor = (tensor - min_value) / ((max_value) - min_value)
        return tensor, min_value.tolist(), max_value.tolist()

def mape(pred, label, eps=1e-2):
    return torch.mean(torch.abs((pred - label) / (label + eps)))

def relative_l2_loss(predict_y, train_label, trunk_num, label_norm, is_threshold=False, k=10):
    predict_y = predict_y.view(-1, trunk_num)
    train_label = train_label.view(-1, trunk_num)
    # Compute the L2 error for each sample
    l2_error = torch.norm(train_label - predict_y, dim=1, p=2)
    # Compute the relative L2 error for each sample
    relative_l2_error = l2_error / label_norm
    mean_relative_l2_error = torch.mean(relative_l2_error)
    if not is_threshold:
        return mean_relative_l2_error
    else:
        top_loss_values, _ = torch.topk(relative_l2_error, k=k)
        # Sum up the top 10 largest values
        sum_top_loss_values = torch.sum(top_loss_values) / k
        return sum_top_loss_values

def train(net, batch_size, max_epoch, TOL,
          data_path, training_sample_name, test_sample_name,
          model_path, model_name, sample_num, train_trunk_sample_num, test_trunk_sample_num,
          device=None, lr=0.001, weight_decay=0, early_stopping_epoch=500,
          is_loss_plot=False, loss_record_interval=50, loss_type="MSE"):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("The device is %s." % device)
    train_dataset = creatDataSet(data_path, training_sample_name, sample_num=sample_num,
                                 trunk_sample_num=train_trunk_sample_num, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    branch_input_min, branch_input_max = train_dataset.branch_input_min, train_dataset.branch_input_max
    trunk_input_min, trunk_input_max = train_dataset.trunk_input_min, train_dataset.trunk_input_max
    output_min, output_max = train_dataset.output_min, train_dataset.output_max

    net.config["branch_input_min"] = branch_input_min
    net.config["branch_input_max"] = branch_input_max
    net.config["trunk_input_min"] = trunk_input_min
    net.config["trunk_input_max"] = trunk_input_max
    net.config["output_min"] = output_min
    net.config["output_max"] = output_max

    test_dataset = creatDataSet(data_path, sample_name=test_sample_name, sample_num=None, trunk_sample_num=test_trunk_sample_num,
                                branch_input_min=branch_input_min, branch_input_max=branch_input_max,
                                trunk_input_min=trunk_input_min, trunk_input_max=trunk_input_max,
                                output_min=output_min, output_max=output_max, device=device)

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    net = net.to(device)

    try:
        net.device_check()
    except:
        pass
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # opt = torch.optim.SGD(net.parameters(), lr=lr)

    loss_fn = torch.nn.MSELoss()
    # loss_fn = mape
    loss_MIN = 1e10

    desc = "start training..."
    pbar = tqdm(range(max_epoch), desc=desc)
    iterCounter = 0

    if is_loss_plot:
        LossEvaluation.init_plot(max_epoch)

    train_trunk_input = train_dataset.trunk_input_tensor
    test_trunk_input = test_dataset.trunk_input_tensor
    train_trunk_sample_num = train_dataset.trunk_sample_num
    test_trunk_sample_num = test_dataset.trunk_sample_num
    for epoch in pbar:
        train_loss_list = []
        for idx, (train_branch_input, train_label, lable_norm) in enumerate(train_loader):
            train_label = train_label.view(-1, train_label.size(2))

            opt.zero_grad()

            predict_y = net.forward_branch_trunk_fixed(branch_input=train_branch_input, trunk_input=train_trunk_input)
            opt.zero_grad()
            if loss_type == "MSE":
                loss = loss_fn(predict_y, train_label)
            elif loss_type == "L2":
                loss = relative_l2_loss(predict_y=predict_y, train_label=train_label, trunk_num=train_trunk_sample_num,
                                        label_norm=lable_norm, is_threshold=False)
            train_loss_list.append(loss.item())
            loss.backward()
            opt.step()

        net.eval()
        train_loss_value = np.mean(train_loss_list)
        if is_loss_plot and epoch % loss_record_interval == 0:
            LossEvaluation.update_loss_train(train_loss_value, epoch)

        for idx, (test_branch_input, test_label, lable_norm) in enumerate(test_loader):
            test_label = test_label.view(-1, test_label.size(2))
            predict_y = net.forward_branch_trunk_fixed(branch_input=test_branch_input,
                                                       trunk_input=test_trunk_input)
            if loss_type == "MSE":
                loss = loss_fn(predict_y, test_label)
            elif loss_type == "L2":
                loss = relative_l2_loss(predict_y=predict_y, train_label=test_label,
                                        trunk_num=test_trunk_sample_num, label_norm=lable_norm)

        test_loss_value = loss.item()
        net.train()

        if is_loss_plot and epoch % loss_record_interval == 0:
            LossEvaluation.update_loss_test(loss.item(), epoch)

        pbar.set_description("Epoch %d, iterCounter %d, Train Loss %.2e, Test Loss %.2e, Minimum Loss %.2e" % (
            epoch, iterCounter, train_loss_value, test_loss_value, loss_MIN))

        if epoch % loss_record_interval == 0 and is_loss_plot and epoch > 0:
            LossEvaluation.update_plot()

        iterCounter += 1

        if loss.item() < loss_MIN:
            iterCounter = 0
            loss_MIN = loss.item()
            if epoch > 1000:
                torch.save(net, model_path + model_name + ".pth")

        if early_stopping_epoch is not None:
            if iterCounter > early_stopping_epoch:
                print("The loss value is not decreasing. Stop training.")
                return

        if loss.item() < TOL:
            print("Tolerance is satisfied! Stop training.")
            torch.save(net, model_path + model_name + ".pth")
            break

    print("The current loss value is %.2e. Stop training." % loss_MIN)
    # NetworkSL.save_model(model, model_path, name)

    return

if __name__ == '__main__':
    pass
