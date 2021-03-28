# image classification on ImageNet

# Requirements

* python packages
* pytorch>=0.4.0
* torchvision>=0.2.1
* tensorboardX
* pyyaml


# Training a model from scratch

* ./train_val.sh ./configs/config_resnet50_2:4.yaml


# Converting the the **NCHW** format in Pytorch to **NHWC** in ASP

|     Model    | Sparse Pattern |    Top1 |         Top5  |   Download |
| ------------ | --- | ---------------|----------|---- |
| ResNet50 |  2:4 | 77.1 |--|[Google Drive](https://drive.google.com/file/d/15uBFc7-7PEpLQg6ZCUnXXVMDOSlCddsx/view?usp=s
haring)|
| ResNet50 |  4:8 | 77.4 |--|[Google Drive](https://drive.google.com/file/d/1PvMsSZJa6ZW1Bva7FfH9zf8ftIt1pYmf/view?usp=s
haring)|


# Results and Model Zoo

|     Model    | Sparse Pattern |    Top1 |         Top5  |   Download |
| ------------ | --- | ---------------|----------|------ |
| ResNet50 |  2:4 | 77.0 |--|[Google Drive](https://drive.google.com/file/d/1zARmlZDI_JWKEteEwNIjcZBGVEiEvWLc/view?usp=sharing)|
| ResNet50 |  1:4 | 75.9 |--|[Google Drive](https://drive.google.com/file/d/1TUvQg4-Y8RdEyTbiuojLWEH64xLLPszG/view?usp=sharing)|
| ResNet50 |  1:4 | 75.9 |--|[Google Drive](https://drive.google.com/file/d/1zARmlZDI_JWKEteEwNIjcZBGVEiEvWLc/view?usp=sharing)|
| ResNet50 |  2:8 | 76.4 |--|[Google Drive](https://drive.google.com/file/d/1zARmlZDI_JWKEteEwNIjcZBGVEiEvWLc/view?usp=sharing)|
| ResNet50 |  4:8 | 77.4 |--|[Google Drive](https://drive.google.com/file/d/1TUvQg4-Y8RdEyTbiuojLWEH64xLLPszG/view?usp=sharing)|
| ResNet18 |  2:4 | 71.2 |--|[Google Drive](https://drive.google.com/file/d/1zARmlZDI_JWKEteEwNIjcZBGVEiEvWLc/view?usp=sharing)|


## Citing 

If you find NM-sparsity and SR-STE useful in your research, please consider citing:

        @inproceedings{zhou2021,
        title={Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch},
        author={Aojun Zhou, Yukun Ma, Junnan Zhu, Jianbo Liu, Zhijie Zhang, Kun Yuan, Wenxiu Sun, Hongsheng Li},
        booktitle={International Conference on Learning Representations},
        year={2021},
        }

