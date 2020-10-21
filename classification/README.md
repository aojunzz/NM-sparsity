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
| ResNet50 |  Dense | 77.3 |--|--|
| ResNet50 |  2:4 | 77.1 |--|--|
