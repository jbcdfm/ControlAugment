# import math
# from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
# from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F, InterpolationMode
import random
import numpy as np
from PIL import Image

class CutOutPIL:
    def __init__(self, mask_size, mask_color=(0, 0, 0)):
        """
        Args:
            mask_size (int): Size of the square mask.
            mask_color (tuple or int): Color to fill the mask. Default is black.
        """
        
        self.mask_size = mask_size
        self.mask_color = mask_color

    def __call__(self, img):

        w, h = img.size
        mask_size_half = self.mask_size // 2

        # Random center for the mask
        cx = random.randint(0, w)
        cy = random.randint(0, h)

        x1 = max(cx - mask_size_half, 0)
        y1 = max(cy - mask_size_half, 0)
        x2 = min(cx + mask_size_half, w)
        y2 = min(cy + mask_size_half, h)

        img_np = np.array(img).copy()
        img_np[y1:y2, x1:x2] = self.mask_color

        return Image.fromarray(img_np)
    


def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[45.*magnitude,0.],
            # shear =[6.3*np.sign(magnitude),0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, 45.*magnitude],
            # shear =[0,6.3*np.sign(magnitude)],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude*img.size[1]/2), 0],
            # translate=[int(np.sign(magnitude)*5), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0,int(magnitude*img.size[0]/2)],
            # translate=[0,int(np.sign(magnitude)*5)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )

    elif op_name == "Scale":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale= 1 + 1/2*magnitude,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )    
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude*60, interpolation=interpolation, fill=fill)       
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1 + 0.9* magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1 +  0.9* magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img,  1 + 0.9* magnitude)
    elif op_name == "Saturation":
        img = F.adjust_saturation(img,  1 + 0.9 * magnitude)
    elif op_name == "Hue":
        img = F.adjust_hue(img, 0.5*magnitude)
    elif op_name == "Solarize":
        img = F.solarize(img,int(255*(1-magnitude*1/2)))
    elif op_name == "Posterize":
        img = F.posterize(img, int(round(8*(1-magnitude*1/2))))
    elif op_name == "AutoContrast":
        img = Image.blend(img,F.autocontrast(img),magnitude) #F.autocontrast(img)
    elif op_name == "Equalize":
        img = Image.blend(img,F.equalize(img),magnitude) #F.equalize(img)
    elif op_name == "CutOut":
        img = CutOutPIL(mask_size = int(magnitude*img.size[0]/2))(img)
    
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

    
     

class SingleAugment(torch.nn.Module):
    r"""Change this text: Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        gamma (float): The augmentation magnitudes of the N augmentation policies. Each value
            is a value between 0 and 1. 
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
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
            "Scale": (gamma, True),
            "Rotate": (gamma, True),
            "Hue": (gamma, True),
            "Brightness": (gamma, True),
            "Sharpness": (gamma, True),
            "Contrast": (gamma, True),
            "Saturation": (gamma, True),
            "Solarize": (gamma,False),
            "Posterize": (gamma,False),
            "AutoContrast": (gamma,False),
            "Equalize": (gamma,False),
            # "CutOut": (gamma,False)
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
        # op_index = int(torch.randint(len(op_meta), (1,)).item())
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
    r"""Dataset-independent data-augmentation with InformedAugment.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        Naugs_geo (int): Number of geometric transforms to be sampled.
        Naugs_app (int): Number of appearance-based transforms to be sampled.
        gamma (float): The maximum augmentation magnitudes of the N operations. Each value
            is a value between 0 and 1. 
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
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

    def _augmentation_space(self) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (signed)
            "TranslateX": (True),
            "TranslateY": (True),
            "ShearX": (True),
            "ShearY": (True),
            "Scale": (True),
            "Rotate": (True),
            "Hue": (True),
            "Brightness": (True),
            "Sharpness": (True),
            "Contrast": (True),
            "Saturation": (True),
            "Solarize": (False),
            "Posterize": (False),
            "AutoContrast": (False),
            "Equalize": (False),
            # "CutOut": (False),
        }
    


    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
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
            if skew[n] == 0:
                magnitude = random.uniform(0,  gamma[index[n]])
            else:
                x = random.uniform(-gamma[index[n]],gamma[index[n]])
                if random.uniform(-gamma[index[n]],gamma[index[n]]) > skew[n]*x:
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
    

