import torch
from PIL import Image
import torch.nn.functional as F


def resize_to_original(tensor, original_size):
    """
    Efficiently resize tensor to original dimensions using GPU interpolation.

    Args:
        tensor: Input tensor of shape (C, H, W)
        original_size: Tuple of (width, height) or tensor containing dimensions

    Returns:
        Resized tensor
    """
    if original_size is None:
        return tensor

    # Handle different original_size formats efficiently
    if isinstance(original_size, (tuple, list)):
        if len(original_size) >= 2:
            # Convert to integers if they're tensors
            width = original_size[0].item() if hasattr(original_size[0], 'item') else original_size[0]
            height = original_size[1].item() if hasattr(original_size[1], 'item') else original_size[1]
            target_size = (height, width)  # (height, width) for interpolate
        else:
            return tensor
    else:
        return tensor

    # Check if resize is needed
    current_size = (tensor.shape[1], tensor.shape[2])  # (H, W)
    if current_size == target_size:
        return tensor

    # Use optimized GPU interpolation
    with torch.no_grad():  # Disable gradients for faster computation
        # Ensure tensor has batch dimension: (C, H, W) -> (1, C, H, W)
        needs_batch_dim = len(tensor.shape) == 3
        if needs_batch_dim:
            tensor = tensor.unsqueeze(0)

        tensor_resized = F.interpolate(
            tensor,
            size=target_size,
            mode='bilinear',
            align_corners=False,
            antialias=True  # Better quality, minimal performance impact
        )

        # Remove batch dimension if we added it
        if needs_batch_dim:
            tensor_resized = tensor_resized.squeeze(0)

    return tensor_resized


def save_image(path, tensor, original_size=None):
    """
    Save tensor as image, optionally resizing to original dimensions.
    Optimized to do resizing on GPU before moving to CPU.

    Args:
        path: Output file path
        tensor: Image tensor
        original_size: Optional tuple of (width, height) to resize to
    """
    # Do resizing on GPU first (much faster than CPU)
    if original_size is not None:
        tensor = resize_to_original(tensor, original_size)

    # Now move to CPU and process
    tensor = tensor.squeeze(0).cpu()
    # Convert to float32 if it's half precision to avoid clamp issues on CPU
    if tensor.dtype == torch.float16:
        tensor = tensor.float()

    tensor = torch.clamp(tensor, 0, 1)

    image = None

    if tensor.size(0) == 3:
        # Convert 3-channel tensor (C, H, W) to (H, W, C)
        tensor = tensor.permute(1, 2, 0).numpy()
        tensor = (tensor * 255).astype('uint8')
        image = Image.fromarray(tensor)  # Create an RGB image
        image.save(path)

    elif tensor.size(0) == 1:
        # Convert 1-channel tensor to grayscale (H, W)
        tensor = tensor.squeeze(0).numpy()
        tensor = (tensor * 255).astype('uint8')
        image = Image.fromarray(tensor, mode='L')  # Create a grayscale image
        image.save(path)
    else:
        pass


def normalize_imgs(rgb, gamma=None, device="cuda", dtype=torch.float32):
    rgb = torch.clamp(rgb, 0, 1)
    if gamma is not None:
        rgb = rgb ** gamma
    rgb_norm: torch.Tensor = rgb * 2.0 - 1.0  # [0, 255] -> [-1, 1]
    rgb_norm = rgb_norm.to(device).to(dtype)
    assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

    return rgb_norm
