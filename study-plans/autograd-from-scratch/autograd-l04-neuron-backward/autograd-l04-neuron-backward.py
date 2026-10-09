import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    import torch

def neuron_backward(inputs, weights, bias, upstream_gradient):
    """
    Returns: output, input gradients, weight gradients, and bias gradient
    """
    x = inputs
    w = weights
    b = bias
    upstream = upstream_gradient
    output = torch.tanh(torch.sum(x * w) + b)
    lg = upstream * (1 - output.square())
    ip = lg * w
    wg = lg * x
    bg = lg
    return (output, ip, wg, bg)

    pass
