import torch
import random
from torchvision import transforms
from torchvision.transforms import v2



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
