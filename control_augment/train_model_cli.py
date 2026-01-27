# -*- coding: utf-8 -*-
"""
Created on Mon _an 20 

@author: JBC
"""

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import time
import multiprocessing
from torch.utils.data import random_split
from torch.utils.data import Subset
from torchvision.transforms import functional as F2, InterpolationMode
from torchvision.transforms import v2
import os
import argparse
import importlib
from datetime import datetime
from src import setup_utils as su
from src import CtrlA_utils as ctrla_utils


# Accuracy Function
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Optimizer update Function
def optimize(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return

# Training Function
def train_model(loader, optimizer, model, criterion, device):
    trn_corr = 0
    trn_loss = 0
    model.train()
    for b, (X_train, y_train) in enumerate(loader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        _, predicted = torch.max(y_pred.data, 1)
        trn_corr += (predicted == y_train).sum().item()
        trn_loss += loss.item()
        optimize(optimizer, loss)
    return trn_corr, trn_loss

# Testing Function
def test_model(loader, model, criterion, device):
    tst_corr = 0
    tst_loss = 0.0
    model.eval()
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(loader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            loss_test = criterion(y_pred, y_test)
            _, predicted = torch.max(y_pred.data, 1)
            tst_corr += (predicted == y_test).sum().item()
            tst_loss += loss_test.item()
            

            
    return tst_corr, tst_loss

def TTA_test_model(test_set, model, criterion, TTA_transforms, batch_sz, num_classes, device):
    
    test_size = len(test_set)
    y_out = torch.zeros([test_size*len(TTA_transforms),num_classes])
    y_true = torch.zeros([test_size,])
    
    for k in range(len(TTA_transforms)):
        test_set.transform = TTA_transforms[k]
        
        test_loader = DataLoader(
            test_set,
            batch_size = batch_sz, 
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            persistent_workers=True,
            )
        
        # test_loader = DataLoader(test_set,batch_size = batch_sz,shuffle=False)
        model.eval()
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                batch_length = len(y_test)
                X_test, y_test = X_test.to(device), y_test.to(device)
                out = nn.functional.softmax(model(X_test), dim=1)
                y_out[k*test_size+b*batch_sz:k*test_size+b*batch_sz+batch_length,:] = out#torch.argmax(out, dim=1)
                if k==0:
                    y_true[b*batch_sz:(b)*batch_sz+batch_length] = y_test
    
    y_pred = torch.argmax(y_out[:test_size], dim=1)              
    test_corr = (y_pred == y_true).sum().item()
    test_acc = test_corr/test_size*100
    
    y_TTA = y_out.view(len(TTA_transforms), test_size, num_classes).mean(dim=0)
    y_pred_TTA = torch.argmax(y_TTA[:test_size], dim=1)      
    test_corr = (y_pred_TTA == y_true).sum().item()
    test_acc_TTA = test_corr/test_size*100
    
                
    return test_acc, test_acc_TTA



# Eval Function for CtrlA dataset
def CtrlA_test_model(test_loader,model, criterion,device):
    model.eval()
    tst_corr = []
    with torch.no_grad():
        for b,(X_test,y_test) in enumerate(test_loader):
            X_test = X_test.to(device); y_test = y_test.to(device)
            y_pred = model.forward(X_test)
            _, predicted = torch.max(y_pred.data, 1)
            tst_corr.append((predicted == y_test).sum().item()) # Get result divided in batches

    return tst_corr



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





def duplicate_and_flip(train_data):
    """
    Args:
        tuple: (image, label): image is type torch, and has shape (N, 3, 32, 32)
    Returns:
        tuple: (image, label) image has shape (2N, 3, 32, 32)
        as every image is horisontally mirrored and stored alongside
        the originals.
    """
    train_data_duplicated = torch.zeros([len(train_data)*2, 3, 32, 32])
    train_labels_duplicated = torch.zeros([len(train_data)*2,])
    for n in range(len(train_data)*2):
        # if n < len(train_data):
        data, label = train_data[n % len(train_data) ]
        if n >= len(train_data):
            data = v2.functional.horizontal_flip(data)
            
        train_data_duplicated[n,:,:,:] = data
        train_labels_duplicated[n] = int(label) 
        
    return  train_data_duplicated, train_labels_duplicated




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

        




# Main Training Loop
def train(N_augs=1, params = {}, dataset = 'cifar10', model_type = 'WideResNet-28-10', val_type = "test_subset", DAtype = 'CtrlA'):
    

    # ASSERTIONS
    assert params['aug_space'] in ["Standard", "Wide", "Control"], f"Invalid value: {params['aug_space']}"
    assert dataset in ["cifar10", "cifar100", "svhn-c"], f"Invalid value: {dataset}"
    assert val_type in ["test_subset", "train_subset"], f"Invalid value: {val_type}"
    assert DAtype in ["CtrlA", "TA"], f"Invalid value: {DAtype}"
    assert params["setup"] in ["standard", "modified"], f"Invalid value: {params['setup']}"
   
    if DAtype == 'CtrlA':
        assert type(N_augs) == int, "N_augs must be an integer for CtrlA"
        N = int(N_augs)
        assert N > 0, "N must be positive or 0" 
        print(f"CtrlA-augmentation: {N_augs}")
        aug = importlib.import_module(f"src.augmentations_CtrlA_{params['aug_space']}")
    elif DAtype == 'TA':
        aug = importlib.import_module("src.augmentations_TA")



    # Choose cuda GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Dataset: {dataset}")
    train_data, test_data, number_classes = su.load_dataset(dataset)
    
    # Model
    model = su.setup_model(model_type,device,number_classes)

    
    val_size = 1000
    # Create splits (with a fixed seed for repeatability)
    if val_type == "train_subset":
        generator = torch.Generator().manual_seed(1)
        train_data, val_data = random_split(
            train_data, [len(train_data)-val_size, val_size], generator=generator)
    elif val_type == "test_subset":
        val_data = Subset(test_data, list(range(val_size)))

    
    
    setup = params["setup"]
    epoch_max = params["nmax"]
    phase_length = params["phase_length"]

    # Create Dataset instance for training data
    if setup == "modified":
        if dataset == 'cifar10' or dataset == 'cifar100':
            new_train_data, new_train_labels = duplicate_and_flip(train_data)
            train_data = Create_train_Dataset(new_train_data,new_train_labels) 
            train_data =  MyDataset(train_data)
            print("Training dataset updated with mirrored versions")
            epoch_max = epoch_max//2
        else:
            train_data =  MyDataset(train_data)

    else:
        train_data =  MyDataset(train_data)
    
    # Create Dataset instance for test data
    test_data =  MyDataset(test_data)  
    
    # Calculate training data per-channel mean and standard deviation
    data_mean, data_std = su.get_mean_and_std(train_data)
    if setup == "modified" and "svhn" in dataset:
        data_mean = (0.5,0.5,0.5)
    print("Data mean:", data_mean)
    print("Data std: ", data_std)
        
    
    val_transform = transforms.Compose([
        transforms.Normalize(data_mean, data_std),
    ])



           
    # Define learning parameters
    lr0 = params["lr"]
    lr_schedule = 1/2*lr0*(1+np.cos(np.pi*np.linspace(0,epoch_max,epoch_max+1)/(epoch_max+1)))        
    wd = params["wd"]
    # Loss criteria
    criterion = nn.CrossEntropyLoss()
    batch_sz = 125  
    
    if setup == "modified": # modified setup uses BILINEAR interpolation
        interp = InterpolationMode.BILINEAR
    else: 
        interp = InterpolationMode.NEAREST
    print(f"Interpolation type in affine augmentations: {interp}")
    
    
    if DAtype == 'CtrlA':    
        transform_vec = list(aug.SingleAugment()._augmentation_space().keys())
        assert list(aug.SingleAugment()._augmentation_space().keys()) == list(aug.ControlAugment()._augmentation_space().keys()), "Augmentation keys don't match for CtrlA"
        
        # Create ControlAugment dataset 
        aug_p_batch = 8
        CtrlA_strengths = 10   # number of gamma values applied to obtain sensitivity curves for each operation
        CtrlA_batch = 125     # 1000/aug_p_batch -> 1000 images per aug strength
        CtrlA_dataset = ctrla_utils.create_CtrlA_test_data(val_data,aug,Naugs=len(transform_vec),Nstrengths=CtrlA_strengths,batch_size=CtrlA_batch, aug_per_batch=aug_p_batch,interp = interp)
        CtrlA_dataset = MyDataset(CtrlA_dataset,transform=val_transform)
        print("ControlAugment dataset created")
        
        # Create Ctrl-A Dataloader
        CtrlA_loader = DataLoader(
            CtrlA_dataset,
            batch_size = CtrlA_batch, 
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            persistent_workers=True,
            )

        # Initiate Ctrl-A parameters
        xi = 0.9# 
        kappa_sp = params["kappa_sp"]            
        Delta_xi_min = 0.005
        Delta_xi_max = 0.1
    
    # After the creation of the Ctrl-A dataset, convert to dataset object:
    val_data =  MyDataset(val_data)   
    

    val_data.transform = val_transform    # Test data transformation
    val_loader = DataLoader(
        val_data,
        batch_size = batch_sz, 
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        persistent_workers=True,
        )
       


    if DAtype == "TA":
        DataAugTransform = aug.TrivialAugment(aug_space=params["aug_space"],interpolation=interp)
        train_transform = su.aug_pipeline(DataAugTransform, dataset, setup, data_mean, data_std)

        train_data.transform = train_transform   # Update training data transformations
        train_loader = DataLoader(
            train_data, 
            batch_size = batch_sz, 
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=6,
            persistent_workers=True,
            )
    
    # Lists updated during training
    train_losses = []
    val_losses = []
    train_correct= []
    val_correct=[]
    arg_strengths = []
    alpha_strengths = []
    kappa = []
    phases = [0]
    
        
    # Initial ASD parameters 
    Gamma = [0.]*len(transform_vec) 
    alpha =  [0.]*len(transform_vec)
    

    i = 1  # epoch index stepper
    j = 1  # phase index stepper
        

    print("Initiating training...")
    train_flag = True
    while train_flag:

        if DAtype == "CtrlA":    # Rerun this every phase to update Gamma and alpha
            DataAugTransform = aug.ControlAugment(gamma = Gamma, skew = alpha, Naugs = N,interpolation=interp)
            train_transform = su.aug_pipeline(DataAugTransform, dataset, setup, data_mean, data_std)
            
            train_data.transform = train_transform   # Update training data transformations
            train_loader = DataLoader(
                train_data, 
                batch_size = batch_sz, 
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                num_workers=6,
                persistent_workers=True,
                )

        if i == 1:
            print("Transform pipeline:")
            print(train_transform)
            start_time = time.time()
        
        print(f"Phase {j} | Learning rate: {lr_schedule[i-1]:.6f}")
        ############ Here Starts Phase j ###########
        phase_flag = True; 
        while phase_flag:
            # Train Model
            lr = lr_schedule[i-1]
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd,nesterov=True)
            
            # CtrlA ASD parameter saving
            arg_strengths.append(Gamma)
            alpha_strengths.append(alpha)
            
            # Train Model
            trn_correct, trn_loss = train_model(train_loader,optimizer,model,criterion,device)
            train_losses.append(trn_loss)
            train_correct.append(trn_correct)
                
            # Evaluate Model
            vl_correct,vl_loss = test_model(val_loader,model,criterion,device)
            val_losses.append(vl_loss)
            val_correct.append(vl_correct)

            
            current_time = time.time()
            print(f"Epoch {i} | {(current_time-start_time)/60:.2f} min | Train loss: {trn_loss/len(train_data)*batch_sz*1000:.2f}m  |  Val. loss: {vl_loss/len(val_data)*batch_sz*1000:.2f}m")
             
            # Here starts the IA procedure
            if DAtype == 'CtrlA':
            
                if i%phase_length == 0 and i<epoch_max:
                    phase_flag = False
                    print(f"Phase {j} ended - finding augmentation strengths for next phase with kappa_sp = {kappa_sp:.2f}")
                    phases.append(i)
                    
                    # Update kappa based on Equations (3) and (4)
                    train_loss_avg = np.mean(np.asarray(train_losses[phases[j-1]:phases[j]]))/len(train_data)
                    val_loss_avg =  np.mean(np.asarray(val_losses[phases[j-1]:phases[j]]))/len(val_data)
                    kappa.append(train_loss_avg/val_loss_avg)
                    
                    
                    if j > 1:
                        # Update xi based on Equations (6)
                        Delta_xi = (1-xi)/2*(kappa[-1]-kappa_sp)
                        if abs(Delta_xi)<Delta_xi_min:
                            Delta_xi =Delta_xi_min*np.sign(Delta_xi)
                        elif abs(Delta_xi)>Delta_xi_max:
                            Delta_xi = Delta_xi_max*np.sign(Delta_xi)

                        xi += Delta_xi
                        if xi < 0:        # Set lower limit of xi
                            xi = 0
                        elif xi > 0.99:   # Set upper limit of xi
                            xi = 0.99

                        print(f"New threshold value, xi: {xi:.3f}, based on kappa_{j}: {kappa[-1]:.2f}")


                    benchmark = np.asarray(val_correct[::-1][0])/len(val_data)
                    print(f"Train Acc.:  {train_correct[-1]/len(train_data)*100:.2f} %, Val. Acc.: {benchmark*100:.2f} %")
                    
                    CtrlA_tst_correct = CtrlA_test_model(CtrlA_loader, model,criterion,device)
                    CtrlA_data = np.mean(np.asarray(CtrlA_tst_correct).reshape(-1, aug_p_batch), axis=1)/CtrlA_batch
                    Gamma, alpha = ctrla_utils.get_ASD(CtrlA_data,len(transform_vec),xi,CtrlA_strengths,Gamma, alpha)     
                    print(f"Gamma: {np.round(Gamma,2)}")
                    print(f"alpha: {np.round(alpha,2)}")
                    j+=1
        
            
            
            if DAtype!="CtrlA" and i%phase_length == 0:
                benchmark = np.asarray(val_correct[::-1][0])/len(val_data)
                print(f"Train Acc.: {train_correct[-1]/len(train_data)*100:.2f} %, Val. Acc.: {benchmark*100:.2f} %,")
                print(f"Learning rate: {lr_schedule[i]:.6f}")
                phases.append(i)
                train_loss_avg = np.mean(np.asarray(train_losses[phases[j-1]:phases[j]]))/len(train_data)
                val_loss_avg =  np.mean(np.asarray(val_losses[phases[j-1]:phases[j]]))/len(val_data)
                kappa.append(train_loss_avg/val_loss_avg)
                j+=1


            if i == epoch_max:
                train_flag = False
                phase_flag = False
                # Compute final kappa value
                phases.append(i)
                train_loss_avg = np.mean(np.asarray(train_losses[phases[j-1]:phases[j]]))/len(train_data)
                val_loss_avg =  np.mean(np.asarray(val_losses[phases[j-1]:phases[j]]))/len(val_data)
                kappa.append(train_loss_avg/val_loss_avg)
            i+=1    
    
    print(f"Training ended after phase {j}" )
    current_time = time.time()
    total = current_time - start_time
    print(f"Training took {total/60} minutes")    
    val_acc = np.asarray(val_correct)/len(val_data)*100
    print(f"Final validation accuracy of {val_acc[::-1][0]} %")


    if "cifar" in dataset:
        TTA_transforms = [transforms.Compose([transforms.Normalize(data_mean, data_std)]),
                      transforms.v2.Compose([transforms.RandomHorizontalFlip(p=1),transforms.Normalize(data_mean, data_std)]),
                      ]
    if "svhn" in dataset: 
        TTA_transforms = [transforms.Compose([transforms.Normalize(data_mean, data_std)]),
                      transforms.v2.Compose([transforms.RandomInvert(p=1),transforms.Normalize(data_mean, data_std)]),
                      ]

    # Evaluate Model
    test_acc, test_acc_TTA = TTA_test_model(test_data, model, criterion, TTA_transforms, batch_sz, number_classes, device)
    
    # Freeing memory
    del model
    torch.cuda.empty_cache()
    
    return test_acc, test_acc_TTA, val_acc, arg_strengths, alpha_strengths, kappa, lr_schedule




def main():
    
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        "--config",
        type=str,
        default="config_cifar10_modified",
        help="Choose which config file from src.configs to use"
        )
    temp_args, _ = parser.parse_known_args()
    cfg = importlib.import_module(f"src.configs.{temp_args.config}")
    
    # Optional config changes:
    parser.add_argument("--dataset", type=str, default=cfg.DATASET)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--model_name", type=str, default=cfg.MODEL_NAME)
    parser.add_argument("--da_type", type=str, default=cfg.DA_TYPE)
    parser.add_argument("--N", type=int, default=cfg.N_AUGS)
    parser.add_argument("--kappa_sp", type=float, default=cfg.KAPPA_SP)
    parser.add_argument("--N_P", type=int, default=cfg.PHASE_LENGTH)
    parser.add_argument("--setup", type=str, default=cfg.SETUP)
    parser.add_argument("--validation_set", type=str, default=cfg.VAL_SET)
    parser.add_argument("--aug_space", type=str, default=cfg.AUG_SPACE)
    
    args = parser.parse_args()

    dict_train = {"kappa_sp": args.kappa_sp,
           "lr": args.learning_rate,
           "wd": args.weight_decay,
           "nmax": args.epochs,
           "n_p": args.N_P,
           "setup": args.setup,
           "aug_space": args.aug_space
           }

    acc, acc_TTA, acc_val, gamma, alpha, kappa, lr = train(N_augs=args.N,
                                                           params = dict_train, 
                                                           model_type = args.model_name, 
                                                           dataset = args.dataset, 
                                                           val_type = args.validation_set, 
                                                           DAtype = args.da_type) # Run model training instance


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
 
