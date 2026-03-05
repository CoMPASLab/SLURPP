import gc
import pickle
import warnings

import numpy.core.multiarray
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def safe_torch_load(path, map_location='cpu'):
    """
    Safely load PyTorch checkpoints with better compatibility.
    Handles numpy compatibility issues and memory optimization.
    """
    # Allow numpy scalar types that appear in legacy checkpoints, then load with
    # weights_only=True for safety.  Falls back to weights_only=False only when
    # the checkpoint contains other unsupported globals.
    try:
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.load(path, map_location=map_location, weights_only=False)


def fix_dtype_mismatch(model, target_dtype=torch.float32):
    """
    Fix dtype mismatches in a model by converting all parameters to the same dtype.
    This resolves issues where some layers have mixed dtypes (e.g., Half and Float).
    """
    print(f"🔧 Converting model to {target_dtype}")

    # Convert all parameters to the target dtype
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)

    # Convert all buffers (including batch norm running stats) to the target dtype
    for name, buffer in model.named_buffers():
        if buffer.dtype != target_dtype:
            buffer.data = buffer.data.to(target_dtype)

    return model


def ensure_consistent_dtype(pipe, use_half_precision=False):
    """
    Ensure all components of the pipeline use consistent dtypes.
    """
    target_dtype = torch.float16 if use_half_precision else torch.float32

    print(f"🔧 Ensuring consistent dtype: {target_dtype}")

    # Fix UNet dtype
    if hasattr(pipe, 'unet') and pipe.unet is not None:
        pipe.unet = fix_dtype_mismatch(pipe.unet, target_dtype)

    # Fix VAE dtype
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        pipe.vae = fix_dtype_mismatch(pipe.vae, target_dtype)

    # Fix text encoder dtype
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        pipe.text_encoder = fix_dtype_mismatch(pipe.text_encoder, target_dtype)

    # Fix VAE CLD if it exists
    if hasattr(pipe, 'vae_cld') and pipe.vae_cld is not None:
        pipe.vae_cld = fix_dtype_mismatch(pipe.vae_cld, target_dtype)

    return pipe


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
    return "CUDA not available"


def enable_memory_efficient_mode(pipe):
    """Enable all available memory efficient settings for the pipeline."""
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ XFormers memory efficient attention enabled")
    except Exception:
        print("⚠ XFormers not available")

    try:
        pipe.enable_attention_slicing()
        print("✓ Attention slicing enabled")
    except Exception:
        print("⚠ Attention slicing not available")

    try:
        pipe.enable_vae_slicing()
        print("✓ VAE slicing enabled")
    except Exception:
        print("⚠ VAE slicing not available")

    try:
        pipe.enable_cpu_offload()
        print("✓ CPU offload enabled")
    except Exception:
        print("⚠ CPU offload not available")


def aggressive_memory_optimization(pipe, device):
    """Apply aggressive memory optimizations."""
    print("🔥 Applying aggressive memory optimizations...")

    # Move all non-essential components to CPU
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        pipe.text_encoder = pipe.text_encoder.to('cpu')
        print("✓ Text encoder moved to CPU")

    if hasattr(pipe, 'vae') and pipe.vae is not None:
        pipe.vae = pipe.vae.to('cpu')
        print("✓ VAE moved to CPU")

    # Keep only UNet on GPU during inference
    if hasattr(pipe, 'unet') and pipe.unet is not None:
        pipe.unet = pipe.unet.to(device)
        print("✓ UNet kept on GPU")

    # Clear cache
    clear_gpu_memory()

    return pipe


def move_component_to_device(component, device):
    """Safely move a component to device when needed."""
    if component is not None:
        component = component.to(device)
        clear_gpu_memory()
    return component


def use_half_precision(pipe):
    """Convert pipeline to half precision to save memory."""
    print("🔧 Converting to half precision (float16)...")

    try:
        if hasattr(pipe, 'unet') and pipe.unet is not None:
            pipe.unet = pipe.unet.half()
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            pipe.vae = pipe.vae.half()
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            pipe.text_encoder = pipe.text_encoder.half()
        if hasattr(pipe, 'vae_cld') and pipe.vae_cld is not None:
            pipe.vae_cld = pipe.vae_cld.half()
        print("✓ Pipeline converted to half precision")
    except Exception as e:
        print(f"⚠ Half precision conversion failed: {e}")

    return pipe


def enable_gradient_checkpointing(pipe):
    """Enable gradient checkpointing to trade compute for memory."""
    print("🔧 Enabling gradient checkpointing...")

    try:
        if hasattr(pipe, 'unet') and pipe.unet is not None:
            pipe.unet.enable_gradient_checkpointing()
            print("✓ UNet gradient checkpointing enabled")
    except Exception as e:
        print(f"⚠ Gradient checkpointing failed: {e}")

    return pipe


def load_pipeline_selective(model_path, low_memory=False):
    """Load pipeline with selective component loading for memory optimization."""
    print(f"🔧 Loading pipeline with low_memory={low_memory}")

    if low_memory:
        # Load components individually to CPU first
        print("Loading components to CPU...")

        # Load UNet to CPU first
        unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", torch_dtype=torch.float32
        )

        # Load VAE to CPU
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.float32
        )

        # Load text encoder to CPU
        text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch.float32
        )

        # Load tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer", clean_up_tokenization_spaces=True
        )

        # Load scheduler
        scheduler = DDIMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        # We'll return the components and let the main script create the pipeline
        return {
            'unet': unet,
            'vae': vae,
            'text_encoder': text_encoder,
            'tokenizer': tokenizer,
            'scheduler': scheduler
        }

    else:
        # Return None to use standard loading
        return None


def reduce_inference_resolution(args, max_size=384):
    """Reduce inference resolution to save memory."""
    if args.inference_resolution > max_size:
        print(f"🔧 Reducing resolution from {args.inference_resolution} to {max_size} to save memory")
        args.inference_resolution = max_size
    return args
