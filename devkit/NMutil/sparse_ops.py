import torch
from torch import autograd


class Sparse(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M):

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, 4)
        index = torch.argsort(weight_temp, dim=1)[:, :N]

        w_b = weight_temp.clone()
        w_b[:] = 1
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        output[w_b==0] = 0

        return output


    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None



class SparseConv(nn.Conv2d):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

       
        setattr(self.weight, "mask", self.mask)



    def get_sparse_weights(self):

        return Sparse.apply(self.weight)



    def forward(self, x):

        w, self.weight.mask = self.get_sparse_weights()

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


