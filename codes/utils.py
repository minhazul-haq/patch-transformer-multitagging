# author: Mohammad Minhazul Haq
# created on: February 3, 2020

import torchvision
import torch.nn as nn


transform_pipe_train = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),  # Convert np array to PILImage

    # randomly change the brightness, contrast and saturation
    torchvision.transforms.ColorJitter(),

    # horizontally flip the given PIL Image randomly with a given probability.
    torchvision.transforms.RandomHorizontalFlip(),

    # vertically flip the given PIL Image randomly with a given probability.
    torchvision.transforms.RandomVerticalFlip(),

    # rotate the image by angle
    torchvision.transforms.RandomRotation(degrees=360),

    # Resize image to 224 x 224 as required by most vision models
    torchvision.transforms.Resize(
        size=(224, 224)
    ),

    # Convert PIL image to tensor with image values in [0, 1]
    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


transform_pipe_val_test = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),  # Convert np array to PILImage

    # Resize image to 224 x 224 as required by most vision models
    torchvision.transforms.Resize(
        size=(224, 224)
    ),

    # Convert PIL image to tensor with image values in [0, 1]
    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
