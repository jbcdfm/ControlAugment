# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 09:05:43 2026

@author: JBC
"""

import random
from src import model_lib
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src import WideResNetModel as WRN
from torch.utils.data import Dataset




class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.dataset)

class Create_train_Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (torch.Tensor): Tensor of images with shape (N, 3, 32, 32)
            labels (torch.Tensor): Tensor of labels with shape (N,)
            transform (callable, optional): Optional transform to apply to the images.
        """
        assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples"
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample and its corresponding label.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (image, label) where image is a transformed tensor (if transform is provided).
        """
        image, label = self.data[idx], self.labels[idx]
        
        if self.transform:
            image = self.transform(image)  # Apply the transform to the image

        label = label.long()

        return image, label


def load_dataset(dataset):

    # Datasets and Data Loaders
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10("/Data/CIFAR10/raw", train=True, download=True,transform=transforms.ToTensor())
        test_data = datasets.CIFAR10("/Data/CIFAR10/raw", train=False, download=True,transform=transforms.ToTensor())
        number_classes = 10

    if dataset == 'cifar100':
        train_data = datasets.CIFAR100("/Data/CIFAR100/raw", train=True, download=True,transform=transforms.ToTensor())
        test_data = datasets.CIFAR100("/Data/CIFAR100/raw", train=False, download=True,transform=transforms.ToTensor())
        number_classes = 100
        
    if dataset == 'svhn-c':
        train_data = datasets.SVHN("/Data/SVHN/raw", split="train", download=True,transform=transforms.ToTensor())
        test_data = datasets.SVHN("/Data/SVHN/raw", split="test", download=True,transform=transforms.ToTensor())
        number_classes = len(np.unique(train_data.labels))        

    return train_data, test_data, number_classes



def setup_model(model_type,device,number_classes):
    # Model
    if model_type == 'LeNet':
        model = model_lib.LeNet().to(device)
    if model_type == 'airbench94':
        model = model_lib.airbench_net().to(device)
    if model_type == 'WideResNet-28-10':
        model = WRN.Wide_ResNet(28,10,dropout_rate=0, num_classes=number_classes).to(device)
        model.apply(WRN.conv_init)

    return model



def get_mean_and_std(train_data):
    loader = DataLoader(train_data, batch_size=100, shuffle=False, num_workers=1)
    sum_rgb = torch.zeros(3)
    sumsq_rgb = torch.zeros(3)
    num_pixels = 0  # total number of pixels per channel
    
    for imgs, _ in loader:          # imgs: (N, C, H, W)
        imgs = imgs.to(torch.float32)
        b, c, h, w = imgs.shape
        num_pixels += b * h * w
        # sum over N,H,W -> result shape (C,)
        sum_rgb += imgs.sum(dim=[0, 2, 3])
        sumsq_rgb += (imgs * imgs).sum(dim=[0, 2, 3])
    
    mean = sum_rgb / num_pixels
    var = (sumsq_rgb / num_pixels) - (mean ** 2)
    std = torch.sqrt(var)
    
    return tuple(mean.tolist()), tuple(std.tolist())


