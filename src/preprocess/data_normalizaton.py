import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMS Normalization Layer
    Normalizes the input tensor based on its root mean square (RMS) value.
    """
    def __init__(self, normalized_shape, eps=1e-8):
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        # Calculate the mean square of the input tensor
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        # Calculate the root mean square (RMS)
        rms = torch.sqrt(mean_square + self.eps)
        # Normalize the input tensor
        x_normalized = x / rms * self.weight

        return x_normalized