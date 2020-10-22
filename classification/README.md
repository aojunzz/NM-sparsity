# image classification on ImageNet

# Requirements

* python packages
* pytorch>=0.4.0
* torchvision>=0.2.1
* tensorboardX
* pyyaml


# Training a model from scratch

* ./train_val.sh configs/config_resnet50_2:4.yaml

# Results and Model Zoo

|     Model    | Sparse Pattern |    Top1 |         Top5  |   Download |
| ------------ | --- | ---------------|----------|------ |
| ResNet50 |  Dense | 77.3 |--|[Google Drive](https://drive.google.com/file/d/1TUvQg4-Y8RdEyTbiuojLWEH64xLLPszG/view?usp=sharing)|
| ResNet50 |  2:4 | 77.1 |--|[Google Drive](https://drive.google.com/file/d/1zARmlZDI_JWKEteEwNIjcZBGVEiEvWLc/view?usp=sharing)|
| ResNet50 |  1:4 | 77.3 |--|[Google Drive](https://drive.google.com/file/d/1TUvQg4-Y8RdEyTbiuojLWEH64xLLPszG/view?usp=sharing)|
| ResNet50 |  2:8 | 77.1 |--|[Google Drive](https://drive.google.com/file/d/1zARmlZDI_JWKEteEwNIjcZBGVEiEvWLc/view?usp=sharing)|
| ResNet50 |  4:8 | 77.3 |--|[Google Drive](https://drive.google.com/file/d/1TUvQg4-Y8RdEyTbiuojLWEH64xLLPszG/view?usp=sharing)|
| ResNet18 |  Dense | 71.3 |--|[Google Drive](https://drive.google.com/file/d/1zARmlZDI_JWKEteEwNIjcZBGVEiEvWLc/view?usp=sharing)|
| ResNet18 |  2:4 | 71.2 |--|[Google Drive](https://drive.google.com/file/d/1zARmlZDI_JWKEteEwNIjcZBGVEiEvWLc/view?usp=sharing)|
