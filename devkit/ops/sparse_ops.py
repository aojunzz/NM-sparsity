import torch
from torch import autograd, nn
import torch.nn.functional as F


class Sparse(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N=2, M=4):

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :N]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)

        return weight*w_b, w_b


    @staticmethod
    def backward(ctx, grad_output, _):
        return grad_output, None



class SparseConv(nn.Conv2d):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)



    def get_sparse_weights(self):

        return Sparse.apply(self.weight)



    def forward(self, x):

        w, mask = self.get_sparse_weights()
        setattr(self.weight, "mask", mask)

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


