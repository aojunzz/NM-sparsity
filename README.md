# N:M Fine-grained Structured Sparse Neural Networks

N:M sparsity is fine-grained structured network, which can maintain the advantages of both unstructured fine-grained sparsity and structured coarse-grained sparsity simultaneously.

For hardware acceleration, you can see the following resources:

&nbsp; [How Sparsity Adds Umph to AI Inference](https://blogs.nvidia.com/blog/2020/05/14/sparsity-ai-inference/)

&nbsp; [Accelerating Sparsity in the NVIDIA Ampere Architecture](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22085-accelerating-sparsity-in-the-nvidia-ampere-architecture%E2%80%8B.pdf)

&nbsp; [Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/) 



## Method

SR-STE can achieve **comparable or even better** results with **negligible extra training cost** and **only a single easy-to-tune hyperparameter $\lambda_w$** than original dense models.


```python
def forward(ctx, weight, N=4, M=2):
    output = weight.clone()
    length = weight.numel()
    group = int(length/M)
    weight_temp = weight.detach().abs().reshape(group, M)
    index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]
    
    # compute the mask ($epsilon_t$ in the paper)
    mask = torch.ones(weight_temp.shape, device=weight_temp.device)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(weight.shape)

    return output*mask, mask
```


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
