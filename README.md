# N:M Fine-grained Structured Sparse Neural Networks
[arxiv](https://arxiv.org/abs/2102.04010), [ICLR2021](https://openreview.net/forum?id=K9bw7vqp_s)

## Why N:M sparsity?

Sparsity in Deep Neural Networks (DNNs) has been widely studied to compress　and accelerate the models on resource-constrained environments. 
It can be generally categorized into unstructured fine-grained sparsity that zeroes out multiple　individual weights distributed across the neural network, and structured coarse-grained sparsity which prunes blocks of a neural network. Fine-grained sparsity can achieve a high compression ratio but is not hardware friendly　and hence receives limited speed gains. On the other hand, coarse-grained sparsity cannot concurrently achieve both acceleration on modern GPUs and maintain performance.

**N:M fine-grained structured** sparse network, which can maintain the advantages of
both unstructured fine-grained sparsity and structured coarse-grained sparsity simultaneously on specifically designed GPUs.




![alt text](NM.png)



Latest NVIDIA Ampere GPUs design for 2:4 sparsity, For hardware acceleration, you can see the following resources:

&nbsp; [How Sparsity Adds Umph to AI Inference](https://blogs.nvidia.com/blog/2020/05/14/sparsity-ai-inference/)

&nbsp; [Accelerating Sparsity in the NVIDIA Ampere Architecture](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22085-accelerating-sparsity-in-the-nvidia-ampere-architecture%E2%80%8B.pdf)

&nbsp; [Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/) 



## Method

SR-STE can achieve **comparable or even better** results with **negligible extra training cost** and **only a single easy-to-tune hyperparameter $\lambda_w$** than original dense models.

![alt text](sr-ste.png)


the implementation details are shown as follows(https://github.com/NM-sparsity/NM-sparsity/blob/main/devkit/sparse_ops/sparse_ops.py):

```python

class Sparse(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b


    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None

```

```python

class SparseConv(nn.Conv2d):
    """" implement N:M sparse convolution layer """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', N=2, M=4, **kwargs):
        self.N = N
        self.M = M
        super(SparseConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)


    def get_sparse_weights(self):

        return Sparse.apply(self.weight, self.N, self.M)

    def forward(self, x):

        w = self.get_sparse_weights()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

```



## Experiments

#### Image Classification on ImageNet 

 [classification](https://github.com/anonymous-NM-sparsity/NM-sparsity/tree/main/classification) 


#### Objection Detection on COCO


 [detection](https://github.com/anonymous-NM-sparsity/NM-sparsity/tree/main/detection) 

#### Instance Segmentation on COCO

 [segmentation](https://github.com/anonymous-NM-sparsity/NM-sparsity/tree/main/classification) 

#### Machine Translation


 [language model](https://github.com/anonymous-NM-sparsity/NM-sparsity/tree/main/classification) 


#### Citing 

If you find NM-sparsity and SR-STE useful in your research, please consider citing:

        @inproceedings{zhou2021,
        title={Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch},
        author={Aojun Zhou, Yukun Ma, Junnan Zhu, Jianbo Liu, Zhijie Zhang, Kun Yuan, Wenxiu Sun, Hongsheng Li},
        booktitle={International Conference on Learning Representations},
        year={2021},
        }
