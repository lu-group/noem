# External Libs.
import torch
import torch.nn as nn
# Internal Libs.
from util.fcnn import BaseNetwork as BaseNetwork
class DeepONet(nn.Module):
    def __init__(self, branchinfo, trunkinfo, channel_size, anainfo=[]):
        super().__init__()
        self.m1_act_fn = branchinfo['act_fn']
        self.m2_act_fn = trunkinfo['act_fn']
        self.m1_input_size = branchinfo['input_size']
        self.m2_input_size = trunkinfo['input_size']
        self.m1_hidden_sizes = branchinfo['hidden_sizes']
        self.m2_hidden_sizes = trunkinfo['hidden_sizes']
        self.m1_output_size = branchinfo['output_size']
        self.m2_output_size = trunkinfo['output_size']
        self.model_channel_size = channel_size
        # model1: branch net
        self.model1 = BaseNetwork(act_fn=self.m1_act_fn, input_size=self.m1_input_size, output_size=self.m1_output_size, hidden_sizes=self.m1_hidden_sizes)
        # model2: trunk net
        self.model2 = BaseNetwork(act_fn=self.m2_act_fn, input_size=self.m2_input_size, output_size=self.m2_output_size, hidden_sizes=self.m2_hidden_sizes)
        self.config = {"branchinfo": branchinfo, "trunkinfo": trunkinfo, "channel_size": channel_size, "anainfo": anainfo}

    def forward(self, x):
        x1 = x[:, 0 : self.m1_input_size]
        x2 = x[:, self.m1_input_size:]
        out1 = self.model1(x1)
        out2 = self.model2(x2)
        used_channel_size = 0
        out = torch.zeros(out1.shape[0], len(self.model_channel_size))
        for i in range(len(self.model_channel_size)):
            tx1 = out1[:, used_channel_size : used_channel_size + self.model_channel_size[i]]
            tx2 = out2[:, used_channel_size : used_channel_size + self.model_channel_size[i]]
            # Dot product of two tensors.
            tout = torch.mul(tx1, tx2)
            summed_tout = torch.sum(tout, dim=1)
            out[:, i] = summed_tout
            used_channel_size += self.model_channel_size[i]
        return out

    def branchnet_output(self, x_br):
        output = self.model1(x_br)
        return output

    def trunk_output(self, x_tr):
        output = self.model2(x_tr)
        return output

    def manual_output(self, output_br, output_tr):
        for i in range(len(self.model_channel_size)):
            tx1 = output_br[:, used_channel_size : used_channel_size + self.model_channel_size[i]]
            tx2 = output_tr[:, used_channel_size : used_channel_size + self.model_channel_size[i]]
            # Dot product of two tensors.
            tout = torch.mul(tx1, tx2)
            summed_tout = torch.sum(tout, dim=1)
            out[:, i] = summed_tout
            used_channel_size += self.model_channel_size[i]
        return out

if __name__ == '__main__':
    branchinfo = {'act_fn': [nn.Tanh(), nn.Tanh(), nn.Tanh()], 'input_size': 3, 'output_size': 10, 'hidden_sizes': [32, 32, 32]}
    trunkinfo = {'act_fn': [nn.Tanh(), nn.Tanh(), nn.Tanh()], 'input_size': 1, 'output_size': 10, 'hidden_sizes': [32, 32, 32]}
    channel_size = [10]
    deeponet = DeepONet(branchinfo, trunkinfo, channel_size)
    torch.save(deeponet, "test.pt")
    print("Save Done!")

    # Load
    model = torch.load("test.pt")
    print("Load Done!")

    x = torch.randn(10, 4)
    out = model(x)
    print(out)