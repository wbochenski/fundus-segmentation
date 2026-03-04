import torch
from PIL import Image
from typing import Tuple, Optional
import torchvision.transforms.functional as TF


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Load an image and convert to RGB format.
    
    Args:
        image_path: Path to the image file
        target_size: Optional (height, width) to resize to
        
    Returns:
        RGB image as torch tensor with shape (3, H, W) and values in [0, 1]
    """
    image = Image.open(image_path).convert('RGB')
    
    # Convert to tensor (3, H, W) with values in [0, 1]
    image_tensor = TF.to_tensor(image)
    
    if target_size is not None:
        # Resize to (3, height, width)
        image_tensor = TF.resize(image_tensor, target_size, interpolation=TF.InterpolationMode.BILINEAR)
    
    return image_tensor


def load_mask(mask_path: str, target_size: Optional[Tuple[int, int]] = None, threshold: float = 0.5) -> torch.Tensor:
    """
    Load a mask and convert to binary format.
    
    Args:
        mask_path: Path to the mask file
        target_size: Optional (height, width) to resize to
        threshold: Threshold for binarization (default: 0.5 for normalized masks)
        
    Returns:
        Binary mask as torch tensor with shape (H, W) and values in {0, 1}
    """
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    
    # Convert to tensor (1, H, W) with values in [0, 1]
    mask_tensor = TF.to_tensor(mask)
    
    if target_size is not None:
        # Resize using nearest neighbor to preserve binary values
        mask_tensor = TF.resize(mask_tensor, target_size, interpolation=TF.InterpolationMode.NEAREST)
    
    # Binarize and remove channel dimension: (1, H, W) -> (H, W)
    binary_mask = (mask_tensor.squeeze(0) > threshold).float()
    
    return binary_mask