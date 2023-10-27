import torch
import torch.nn as nn
from torch import Tensor

# param dim: Number of input/output channels
# param n div: Reciprocal of the partial ratio.
# param forward: Forward type, can be either 'split_cat' or 'slicing':
# param kernel size: Kernel size.
class PConv(nn.Module):
    def __init__(self, dim: int, n_div: int, forward: str = "split_cat", kernel_size: int = 3) -> None:
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv

        self.conv = nn.Conv2d(
            self.dim_conv,
            128,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False)

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        x[:, :self.dim.conv, :, :] = self.conv(x[:, :self.dim.conv, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1,x2 = torch.split(x,[self.dim_conv, self.dim_untouched], dim = 1)
        x1 = self.conv(x1)
        x = torch.cat((x1,x2),1)
        return x
