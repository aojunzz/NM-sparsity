import torch
from torch import autograd, nn
import torch.nn.functional as F

from itertools import repeat
from torch._six import container_abcs


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)



class Sparse(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M):

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)

        return output*w_b, w_b


    @staticmethod
    def backward(ctx, grad_output, _):
        return grad_output, None, None

class SparseConv(nn.Conv2d):
    

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=4, **kwargs):
        self.N = N
        self.M = M
        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)


    def get_sparse_weights(self):

        return Sparse.apply(self.weight, self.N, self.M)



    def forward(self, x):

        w, mask = self.get_sparse_weights()
        setattr(self.weight, "mask", mask)
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

#class SparseLinear(nn.Linear):
#    def __init__():
    

    


