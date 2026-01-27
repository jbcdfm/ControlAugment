# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 09:47:26 2026

@author: JBC
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F2, InterpolationMode
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset

from scipy.special import erf
from scipy.optimize import curve_fit



class Create_Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (torch.Tensor): Tensor of images with shape (N, 3, 32, 32)
            labels (torch.Tensor): Tensor of labels with shape (N,)
            transform (callable, optional): Optional transform to apply to the images.
        """
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


        return image, label


class PIL2Tensor_dataset(Dataset):
    def __init__(self,pil_dataset):
        self.pil_dataset = pil_dataset
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.pil_dataset)
    
    def __getitem__(self,idx):
        img,label = self.pil_dataset[idx]
        
        img = self.transform(img)
        
        return img,label
        
    



def erf_fit(x,x0,A,B,dx):
    return B - A*(erf((x-x0)/(dx)))



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def create_CtrlA_test_data(test_data,aug,Naugs=15,Nstrengths=10,batch_size=125,aug_per_batch=8, interp=InterpolationMode.BILINEAR):
    
    magnitude_vec = [(x+1)/Nstrengths for x in range(Nstrengths)]# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]    
    ind_min = 0
    ind_max = batch_size*aug_per_batch
    
    pil_testdata = [(transforms.ToPILImage()(img),label) for img,label in test_data]
    
    
    # Create benchmark (not transformed data)
    subdata = Subset(pil_testdata, indices=range(ind_min,ind_max))
    data = [image for (image,_) in subdata]
    label =  [label for (_,label) in subdata]
    CtrlA_dataset = Create_Dataset(data,label)
    
    
    # Create augmented version of the same data
    for m in range(Naugs):
        transformtype = list(aug.SingleAugment()._augmentation_space())[m]
        signed = aug.SingleAugment()._augmentation_space()[transformtype][1]
        # print(f"Transform type: {transformtype}")
        for n in range(Nstrengths):
            magnitude = magnitude_vec[n]
            subdata = Subset(pil_testdata, indices=range(ind_min,ind_max))
            
            if signed:
                data = [aug._apply_op(image,transformtype,magnitude*(-1)**i,interpolation=interp, fill=None) for i, (image,_) in enumerate(subdata)]
                label = [label for (_,label) in subdata]
            else:
                data = [aug._apply_op(image,transformtype,magnitude,interpolation=interp, fill=None) for (image,_) in subdata]
                label = [label for (_,label) in subdata]
            
            CtrlA_subset = Create_Dataset(data,label)  
            CtrlA_dataset = ConcatDataset([CtrlA_dataset,CtrlA_subset])
                
            
    CtrlA_dataset = PIL2Tensor_dataset(CtrlA_dataset)
            
    return CtrlA_dataset



def get_ASD(CtrlA_correct, Naugs, xi, Nstrengths,Gamma_prev, alpha_prev):
    

    Gamma = []   
    alpha = []
    benchmark = CtrlA_correct[0] # The first value is the benchmark (performance without transforms)
    
    OSC = np.ones([Naugs,Nstrengths+1])
    OSC[:,1:] = np.reshape(np.asarray(CtrlA_correct[1:]),[Naugs,Nstrengths])/benchmark
    gamma_ = np.linspace(0,1,Nstrengths+1)
    gamma_dense = np.linspace(0,1,1001)
    for j in range(Naugs):       
        try:
            if OSC[j,-1] > OSC[j,0]*xi:
                Gamma.append(1)
                alpha.append(min(1,(OSC[j,-1]-xi)/(1-xi)))
            else:
                popt, pcov = curve_fit(erf_fit,gamma_,OSC[j,:], p0 = [0.4,0.4, 0.5,0.3], ftol=0.0001, xtol=0.0001,maxfev=10000)
                A = erf_fit(gamma_dense,*popt)
                idx = find_nearest(A/A[0],xi)
                if gamma_dense[idx]==0 or popt[1]*popt[3]<0:
                    Gamma.append(1)
                    alpha.append(1)
                    # print(IDA_curves[j,:])
                else:        
                    Gamma.append(gamma_dense[idx])
                    alpha.append(0)
                    
        except:
            Gamma.append(Gamma_prev[j])
            alpha.append(alpha_prev[j])
        
    
    return Gamma, alpha
