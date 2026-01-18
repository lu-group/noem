import torch

def gradients(U, X, order=1, device=torch.device('cpu')):
    if order == 1:
        return torch.autograd.grad(U, X, grad_outputs=torch.ones_like(U),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True,)[0].to(device)
    else:
        return gradients(gradients(U, X), X, order=order - 1).to(device)