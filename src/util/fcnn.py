import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self, act_fn, input_size=3, output_size=2, hidden_sizes=[32,32,32], anainfo=[], input_scaler=None, output_scaler=None):
        """
        Inputs:
            act_fn - Object of the activation function that should be used as non-linearity in the network.
            input_size - Size of the input
            output_size - Size of the output
            hidden_sizes - A list of integers specifying the hidden layer sizes in the NN
        """
        super().__init__()
        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        if type(act_fn) != type([]):
            tact_fn = act_fn
            act_fn = []
            for ii in range(len(layer_sizes) - 1):
                act_fn.append(tact_fn)
        for layer_index in range(1, len(layer_sizes)):
            layers += [nn.Linear(layer_sizes[layer_index-1], layer_sizes[layer_index]),
                       act_fn[layer_index - 1]]
        layers += [nn.Linear(layer_sizes[-1], output_size)]
        self.layers = nn.Sequential(*layers) # nn.Sequential summarizes a list of modules into a single module, applying them in sequence
        # We store all hyperparameters in a dictionary for saving and loading of the model
        act_fn_name = []
        for ii in act_fn:
            act_fn_name.append(ii._get_name())
        self.config = {"act_fn": act_fn_name,"input_size": input_size, "output_size": output_size,
                       "hidden_sizes": hidden_sizes, "anainfo": anainfo}

    def forward(self, x):
        out = self.layers(x)
        return out

if __name__ == '__main__':
    BaseNetwork(nn.Tanh())
