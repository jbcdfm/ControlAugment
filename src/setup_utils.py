# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 09:05:43 2026

@author: JBC
"""

import random
import model_lib
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from WideResNetModel import Wide_ResNet
import WideResNetModel



class CutOut:
    def __init__(self, mask_size, mask_color=(0.0, 0.0, 0.0)):
        """
        Args:
            mask_size (int): Size of the square mask.
            mask_color (tuple of float): RGB values for the mask (length must match C)
        """
        self.mask_size = mask_size
        self.mask_color = mask_color

    def __call__(self, img):
        """
        Args:
            img (Tensor): Single image (C,H,W)
        Returns:
            Tensor: Image with random square masked out
        """
        if not torch.is_tensor(img):
            raise TypeError("Input should be a torch.Tensor")
        
        C, H, W = img.shape
        if len(self.mask_color) != C:
            raise ValueError(f"mask_color length {len(self.mask_color)} does not match number of channels {C}")

        # Random center
        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)

        mask_half = self.mask_size // 2
        x1, x2 = max(cx - mask_half, 0), min(cx + mask_half, W)
        y1, y2 = max(cy - mask_half, 0), min(cy + mask_half, H)

        # Apply mask per channel
        for i in range(C):
            img[i, y1:y2, x1:x2] = self.mask_color[i]

        return img
    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f" Mask size: {self.mask_size}"
            f", Mask color: {self.mask_color}"
            f")"
        )
        return s  




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
        model = Wide_ResNet(28,10,dropout_rate=0, num_classes=number_classes).to(device)
        model.apply(WideResNetModel.conv_init)

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



def aug_pipeline(DataAugTransform, dataset, setup, data_mean, data_std):
    

    if dataset == 'cifar10':
        if setup == "modified":  # our setup for CIFAR, horizontal flips embedded
            train_transform = transforms.Compose([
                transforms.RandomCrop(32,padding=4,padding_mode='edge'),
                transforms.ToPILImage(),
                DataAugTransform,   
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
            ])
        else: #else use standard pipelline
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32,padding=4,padding_mode='edge'),
                transforms.ToPILImage(),
                DataAugTransform,
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
                CutOut(16),
            ])
            
    if dataset == 'cifar100':
        if setup == "modified":  # our setup for CIFAR, horizontal flips embedded
            train_transform = transforms.Compose([
                transforms.RandomCrop(32,padding=4,padding_mode='edge'),
                transforms.ToPILImage(),
                DataAugTransform,
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
                CutOut(16),
            ])
        else: #else use standard pipelline
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32,padding=4,padding_mode='edge'),
                transforms.ToPILImage(),
                DataAugTransform,
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
                CutOut(16),
            ])
            
    if dataset == 'svhn-c':
        if setup == "modified": # our setup for SVHN, inverted images embedded
            train_transform = transforms.Compose([
                transforms.RandomInvert(p=0.5),
                transforms.ToPILImage(),
                DataAugTransform,
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
                CutOut(10),
            ])
        else: #else use standard pipelline
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                DataAugTransform,
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std),
                CutOut(16),
            ])
 
    
    return train_transform