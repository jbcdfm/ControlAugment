from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torchvision.transforms import functional as F, InterpolationMode
import random
from PIL import Image


def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[45.*magnitude,0.],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, 45.*magnitude],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude*img.size[1]), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0,int(magnitude*img.size[0])],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )


    elif op_name == "Rotate":
        img = F.rotate(img, magnitude*135, interpolation=interpolation, fill=fill)
        
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1 + 0.99* magnitude)

    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1 +  0.99* magnitude)
      
    elif op_name == "Contrast":
        img = F.adjust_contrast(img,  1 + 0.99* magnitude)

    elif op_name == "Saturation":
        img = F.adjust_saturation(img,  1 + 0.99 * magnitude)

    elif op_name == "Solarize":
        img = F.solarize(img,int(255*(1-magnitude)))
    elif op_name == "Posterize":
        img = F.posterize(img, int(round(8*(1-magnitude*3/4))))
    elif op_name == "AutoContrast":
        img = Image.blend(img,F.autocontrast(img),magnitude) #F.autocontrast(img)
    elif op_name == "Equalize":
        img = Image.blend(img,F.equalize(img),magnitude) #F.equalize(img)

        
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

    
     

class SingleAugment(torch.nn.Module):
    r"""Performs a single data transform depending on operation index and magnitude.    
    This class is used to create the ControlAugment validation dataset. 
    
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Partly adapted from the torchvision.transforms.TrivialAugmentWide module.


    Args:
        gamma (float): The augmentation magnitudes of the K augmentation policies. Each element
            is a value between 0 and 1. 
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is set to ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        op_index: int = 0,
        gamma: list[float] = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.op_index = op_index
        self.gamma = gamma
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self) -> Dict[str, Tuple[Tensor, bool]]:
        gamma = self.gamma
        return {
            # op_name: (magnitudes, signed)
            "TranslateX": (gamma, True),
            "TranslateY": (gamma, True),
            "ShearX": (gamma, True),
            "ShearY": (gamma, True),
            "Rotate": (gamma, True),
            "Brightness": (gamma, True),
            "Sharpness": (gamma, True),
            "Contrast": (gamma, True),
            "Saturation": (gamma, True),
            "Solarize": (gamma,False),
            "Posterize": (gamma,False),
            "AutoContrast": (gamma,False),
            "Equalize": (gamma,False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        op_index = self.op_index
        
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space()
        op_name = list(op_meta.keys())[op_index]
        magnitude, signed = op_meta[op_name]
        
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f", op_index={self.op_index}"
            f", gamma={self.gamma}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s  
    




class ControlAugment(torch.nn.Module):
    r"""Data augmentation pipeline in the ControlAugment implementation based on
    adaptable data augmentation strength distributions. The class receives three inputs: 
        - the number of operations to be sampled in each image instance,
        - the maximum augmentation strength of each transformation type (Gamma)
        - the augmentation-strength distribution skewness for each transformation type (alpha)

    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Partly adapted from the torchvision.transforms.TrivialAugmentWide module.



    Args:
        Naugs (int): Number of transforms to be sampled.
        gamma (float): The maximum augmentation magnitudes of the K operations. Each element
            is a value between 0 and 1. 
        skew (float): The degree of distribution skew of the K operations. Each element
            is a value between 0 and 1.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        Naugs: int = 1,
        gamma: list[float] = None,
        skew: list[float] = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.Naugs = Naugs
        self.gamma = gamma
        self.skew = skew
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self) -> Dict[str, bool]:
        return {
            # op_name: (signed)
            "TranslateX": (True),
            "TranslateY": (True),
            "ShearX": (True),
            "ShearY": (True),
            "Rotate": (True),
            "Brightness": (True),
            "Sharpness": (True),
            "Contrast": (True),
            "Saturation": (True),
            "Solarize": (False),
            "Posterize": (False),
            "AutoContrast": (False),
            "Equalize": (False),
        }


    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Transformed image.
        """
        fill = self.fill  
        gamma = self.gamma
        skew = self.skew
        Naugs = self.Naugs
        op_len = len(self._augmentation_space())
    

        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]


        
        index = random.sample(range(op_len), Naugs)
        


        op_meta = self._augmentation_space()
        for n in range(len(index)):
            op_name = list(op_meta.keys())[index[n]]
            signed = op_meta[op_name]
            if skew[index[n]] == 0:  # sample from uniform distribution
                magnitude = random.uniform(0,  gamma[index[n]])
            else:  # sample from skewed distribution
                x = random.uniform(-gamma[index[n]],gamma[index[n]])
                if random.uniform(-gamma[index[n]],gamma[index[n]]) > skew[index[n]]*x:
                    x = -x
                magnitude = x/2 + gamma[index[n]]/2
            
            
            
            
            if signed and random.randint(0,1):
                magnitude *= -1.0
                
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"N={self.Naugs}"
            f", Gamma={self.gamma}"
            f", alpha={self.skew}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s  
    

