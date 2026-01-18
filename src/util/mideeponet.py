# External Libs.
import torch
import torch.nn as nn
# Internal Libs.
from src.util.fcnn import BaseNetwork as BaseNetwork
class MIDeepONet(nn.Module):
    def __init__(self, branchinfo_list, trunkinfo, channel_size, anainfo=[]):
        super().__init__()
        # initialize branch networks
        self.branchnet_list = []
        for branchinfo in branchinfo_list:
            tbranchnet_act_fn = branchinfo['act_fn']
            tbranchnet_input_size = branchinfo['input_size']
            tbranchnet_hidden_sizes = branchinfo['hidden_sizes']
            tbranchnet_output_size = branchinfo['output_size']
            tbranchnet = BaseNetwork(act_fn=tbranchnet_act_fn, input_size=tbranchnet_input_size, output_size=tbranchnet_output_size, hidden_sizes=tbranchnet_hidden_sizes)
            self.branchnet_list.append(tbranchnet)
        # initialize trunk network
        self.trunknet_act_fn = trunkinfo['act_fn']
        self.trunknet_input_size = trunkinfo['input_size']
        self.trunknet_hidden_sizes = trunkinfo['hidden_sizes']
        self.trunknet_output_size = trunkinfo['output_size']
        self.trunk_net = BaseNetwork(act_fn=self.trunknet_act_fn, input_size=self.trunknet_input_size, output_size=self.trunknet_output_size, hidden_sizes=self.trunknet_hidden_sizes)

        self.model_channel_size = channel_size
        # store the configuration
        self.config = {"branchinfo": branchinfo_list, "trunkinfo": trunkinfo, "channel_size": channel_size, "anainfo": anainfo}

    def forward(self, x):
        split_sizes = [info['input_size'] for info in self.config['branchinfo']]
        split_sizes.append(self.trunknet_input_size)
        inputs = torch.split(x, split_sizes, dim=1)
        branch_inputs = inputs[:-1]
        trunk_input = inputs[-1]

        branch_outputs = [branchnet(branch_input) for branchnet, branch_input in
                          zip(self.branchnet_list, branch_inputs)]
        trunk_output = self.trunk_net(trunk_input)

        out = self.batch_segmented_dot_product(branch_outputs, trunk_output, self.model_channel_size)
        return out

    def batch_segmented_dot_product(self, branch_output_list, trunk_output, segment_sizes):
        output_list = branch_output_list + [trunk_output]
        stacked_tensors = torch.stack(output_list)
        mul_tensors = torch.prod(stacked_tensors, dim=0)
        # Sum the mul_tensors at each row according to the segment_sizes
        results = torch.zeros(len(trunk_output), len(segment_sizes), device=trunk_output.device)
        start_idx = 0
        for j in range(len(segment_sizes)):
            length = segment_sizes[j]
            results[:, j] = mul_tensors[:, start_idx:start_idx + length].sum(dim=1)
            start_idx += length
        return results

    def forward_branch_trunk_fixed(self, branch_input_list, trunk_input):
        # Compute the output for each branch network on its respective input
        branch_outputs = [self.branchnet_list[i](input) for i, input in enumerate(branch_input_list)]

        # Compute the trunk network output
        trunk_output = self.trunk_net(trunk_input)

        # The branch outputs should be expanded along the batch dimension to match the trunk output size
        expanded_branch_outputs = []
        trunk_input_num = len(trunk_input)
        for branch_output in branch_outputs:
            # Expand each branch output to match the number of samples in the trunk output
            expanded_output = branch_output.repeat_interleave(trunk_input_num, dim=0)
            expanded_branch_outputs.append(expanded_output)

        # Trunk output needs to be repeated for each branch output entry
        expanded_trunk_output = trunk_output.repeat(len(branch_input_list[0]), 1)

        # Use batch_segmented_dot_product to compute the dot products
        out = self.batch_segmented_dot_product(expanded_branch_outputs, expanded_trunk_output, self.model_channel_size)
        return out

    def device_check(self):
        for i in range(len(self.branchnet_list)):
            self.branchnet_list[i].to(next(self.trunk_net.parameters()).device)

if __name__ == '__main__':
    pass
