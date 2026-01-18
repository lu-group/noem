import torch

def gradients(U, X, order=1):
    if order == 1:
        return torch.autograd.grad(U, X, grad_outputs=torch.ones_like(U),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True,)[0]
    else:
        return gradients(gradients(U, X), X, order=order - 1)


def gradients_2(U, X, order=1):
    if order == 1:
        # Initialize the gradient result with zeros
        grads = torch.zeros_like(U)
        for i in range(len(U)):
            # Create a vector for grad_outputs where only the i-th element is 1
            grad_outputs = torch.zeros_like(U)
            grad_outputs[i] = 1
            # Compute the gradient for each component of U
            grads[i] = torch.autograd.grad(U, X, grad_outputs=grad_outputs,
                                           create_graph=True,
                                           retain_graph=True,
                                           only_inputs=True)[0]
        return grads
    else:
        return gradients(gradients(U, X), X, order=order - 1)
if __name__ == '__main__':
    x = torch.tensor([[1.0],[3],[9]], requires_grad=True)
    y = x ** 2 + 1
    dy_dx = gradients(y, x, order=2)
    print(dy_dx)
