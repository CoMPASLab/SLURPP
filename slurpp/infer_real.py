# this code is modified from Marigold: https://github.com/prs-eth/Marigold

import argparse
import logging
import os
import sys

import torch
from datasets.UR_real_data import UnderwaterRealDataset
from src.util.config_util import recursive_load_config
from stage2 import CrossLatentUNet
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from slurpp import load_stage1
from slurpp.io import normalize_imgs, save_image

# Add parent directory to path for memory_utils
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
try:
    from slurpp.memory_utils import (get_memory_usage, safe_torch_load, ensure_consistent_dtype,
                                     reduce_inference_resolution, use_half_precision, enable_gradient_checkpointing,
                                     enable_memory_efficient_mode, clear_gpu_memory)
    MEMORY_UTILS_AVAILABLE = True
except ImportError:
    MEMORY_UTILS_AVAILABLE = False
    print("Memory utils not available - running without memory optimizations")

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image underwater restoration with SLURPP."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="path to the data directory containg underwater images.",
    )

    parser.add_argument(
        "--stage2_checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )

    parser.add_argument(
        "--inference_resolution",
        type=int,
        default=512,  # quantitative evaluation uses 50 steps
        help="Resolution for inference, default is 512.",
    )

    # Memory optimization arguments
    parser.add_argument(
        "--low_memory",
        action="store_true",
        help="Enable low memory mode with CPU offloading.",
    )

    parser.add_argument(
        "--disable_text_encoder",
        action="store_true",
        help="Disable text encoder to save memory (for unconditional generation).",
    )

    parser.add_argument(
        "--disable_stage2",
        action="store_true",
        help="Disable stage2 processing to save memory.",
    )

    parser.add_argument(
        "--offload_unet",
        action="store_true",
        help="Move UNet to CPU when not in use (saves most memory but slower).",
    )

    parser.add_argument(
        "--restore_original_size",
        action="store_true",
        help="Resize output images back to original input dimensions.",
    )

    parser.add_argument(
        "--full_save",
        action="store_true",
        help="Save all output images (ill, bc, clear, composite). By default, only saves the corrected image.",
    )

    args = parser.parse_args()
    cfg = args.config
    cfg = recursive_load_config(args.config)
    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps

    # -------------------- Preparation --------------------
    # Random seed
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # Print initial memory usage
    if MEMORY_UTILS_AVAILABLE and device.type == 'cuda':
        print(f"Initial GPU memory: {get_memory_usage()}")

    # -------------------- Data --------------------
    image_size = args.inference_resolution
    data_dir = args.data_dir
    val_ds = UnderwaterRealDataset(root_dir=data_dir, image_size=image_size)
    dataloader = DataLoader(val_ds, num_workers=1, batch_size=1, shuffle=False)

    # -------------------- Model --------------------
    dtype = torch.float32
    base_ckpt_dir = os.environ["BASE_CKPT_DIR"]
    model_path = f"{base_ckpt_dir}/stable-diffusion-2"

    print("LOADING STAGE 1")

    # Check if low memory mode is requested
    low_memory_mode = hasattr(args, 'low_memory') and args.low_memory

    pipe, inputs_fields, outputs_fields, dual = load_stage1(
        model_path, checkpoint_path, cfg, low_memory=low_memory_mode
    )

    stage2 = args.stage2_checkpoint is not None
    if stage2:
        model = CrossLatentUNet(config_path=f"{model_path}/vae/config.json")
        checkpoint_path = args.stage2_checkpoint
        # Load checkpoint with safe loading method
        if MEMORY_UTILS_AVAILABLE:
            checkpoint = safe_torch_load(checkpoint_path, map_location='cpu')
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

        # Only move to GPU if not in low memory mode or if stage2 is not disabled
        if not (MEMORY_UTILS_AVAILABLE and args.low_memory):
            model = model.to(device)
        else:
            model = model.to('cpu')  # Keep on CPU for dynamic loading
            print("✓ Stage2 model kept on CPU for dynamic loading")

        # Fix dtype consistency for Stage2 model
        if MEMORY_UTILS_AVAILABLE:
            model = ensure_consistent_dtype(model, use_half_precision=args.offload_unet if args.offload_unet else False)

        print(f"          ===> Checkpoint Loaded From: {checkpoint_path} ...")
        del checkpoint  # Free memory immediately
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache if using CUDA
        # faster inference
        pipe.skip_connection = True
        pipe.vae_cld = model
        stage2 = False  # in this case stage2 inference is integrated into stage1
    else:
        print("          ===> No stage2 checkpoint provided ...")

    print(f"inputs_fields {inputs_fields}")
    print(f"outputs_fields {outputs_fields}")

    # Apply memory optimizations BEFORE moving to device
    if MEMORY_UTILS_AVAILABLE:
        print(f"Initial memory usage: {get_memory_usage()}")

        # Reduce resolution if low memory mode
        if args.low_memory:
            args = reduce_inference_resolution(args, max_size=384)
            print(f"Resolution reduced to {args.inference_resolution} for memory optimization")

        # Remove components before loading to GPU
        if args.disable_text_encoder and hasattr(pipe, 'text_encoder'):
            del pipe.text_encoder
            pipe.text_encoder = None
            print("✓ Text encoder removed before GPU loading")

        if args.disable_stage2 and hasattr(pipe, 'vae_cld') and pipe.vae_cld is not None:
            del pipe.vae_cld
            pipe.vae_cld = None
            pipe.skip_connection = False
            print("✓ Stage2 model removed before GPU loading")

    # Only move UNet to GPU initially for low memory mode
    if MEMORY_UTILS_AVAILABLE and args.low_memory:
        print("🔧 Applying aggressive memory optimization - selective GPU loading")

        # Only move UNet to GPU
        if hasattr(pipe, 'unet') and pipe.unet is not None:
            pipe.unet = pipe.unet.to(device)
            print("✓ UNet moved to GPU")
            if MEMORY_UTILS_AVAILABLE:
                print(f"   Memory: {get_memory_usage()}")

        # Keep VAE and text encoder on CPU
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            pipe.vae = pipe.vae.to('cpu')
            print("✓ VAE kept on CPU")

        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            pipe.text_encoder = pipe.text_encoder.to('cpu')
            print("✓ Text encoder kept on CPU")

        # Handle Stage2 model
        if hasattr(pipe, 'vae_cld') and pipe.vae_cld is not None:
            pipe.vae_cld = pipe.vae_cld.to('cpu')
            print("✓ Stage2 model kept on CPU")

        # Also move scheduler to appropriate device (usually CPU is fine)
        if hasattr(pipe, 'scheduler'):
            # Scheduler doesn't need to be on GPU
            pass

        if MEMORY_UTILS_AVAILABLE:
            print(f"Memory after selective assignment: {get_memory_usage()}")

        # Apply half precision if requested
        if args.offload_unet:
            pipe = use_half_precision(pipe)
            pipe = enable_gradient_checkpointing(pipe)
            if MEMORY_UTILS_AVAILABLE:
                print(f"Memory after half precision: {get_memory_usage()}")

    else:
        # Standard loading - move everything to device
        pipe = pipe.to(device)
        if MEMORY_UTILS_AVAILABLE:
            print(f"Memory after moving everything to GPU: {get_memory_usage()}")

    # Fix dtype mismatches to prevent Half/Float errors
    if MEMORY_UTILS_AVAILABLE:
        pipe = ensure_consistent_dtype(pipe, use_half_precision=args.offload_unet if args.offload_unet else False)

    # Apply additional memory optimizations
    if MEMORY_UTILS_AVAILABLE:
        # Enable memory efficient mode
        enable_memory_efficient_mode(pipe)
        print(f"Memory after all optimizations: {get_memory_usage()}")
    else:
        # Fallback memory optimizations
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            logging.debug("run without xformers")

        # Enable memory efficient attention for better memory usage
        try:
            pipe.enable_attention_slicing()
        except Exception:
            logging.debug("attention slicing not available")

        # Enable VAE slicing to reduce memory usage
        try:
            pipe.enable_vae_slicing()
        except Exception:
            logging.debug("VAE slicing not available")

    one_step = getattr(cfg, 'one_step', False)
    if one_step:
        print("using one step inference")
        denoise_steps = 1
    if denoise_steps == 1:
        pipe.scheduler.config.timestep_spacing = 'trailing'

    for epoch in range(1):
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(dataloader, disable=None)):

                count = batch_id
                save_to_dir = f"{output_dir}/"

                os.makedirs(save_to_dir, exist_ok=True)

                inputs = []
                for field in inputs_fields:
                    # Keep inputs on CPU until needed to save GPU memory
                    inputs.append(batch["imgs"][field])
                save_name = batch["imgs"]["name"][0]
                # Get original image dimensions for resizing output
                original_size = batch["imgs"].get("original_size", None)
                if original_size is not None:
                    # original_size is a tuple in the batch, extract the first item (for batch size 1)
                    if hasattr(original_size, '__len__') and len(original_size) > 0:
                        original_size = original_size[0] if len(original_size) == 1 else original_size
                    # If it's still a tensor, convert to tuple of integers
                    if hasattr(original_size, 'tolist'):
                        original_size = tuple(original_size.tolist())
                    elif hasattr(original_size, '__iter__') and not isinstance(original_size, (str, bytes)):
                        # Handle case where we have tensors in a tuple: (tensor([4056]), tensor([3040]))
                        if hasattr(original_size[0], 'item'):
                            original_size = tuple(item.item() for item in original_size)
                        else:
                            original_size = tuple(original_size)

                for i in range(len(inputs)):
                    inputs[i] = normalize_imgs(inputs[i], device=device)

                output_pred = pipe(
                    inputs,
                    denoising_steps=denoise_steps,
                    show_progress_bar=False,
                    return_latent=True,
                    is_dual=dual,
                )

                output_pred_latent = output_pred[1]
                output_pred_stage_1 = output_pred[0]
                if stage2:
                    output_pred = output_pred_stage_1
                    composite_img = inputs[0]
                    output_pred_stage_2 = model(output_pred_latent[:, :4], composite_img)
                    output_pred[0:1] = (output_pred_stage_2 / 2 + 0.5).clamp(0, 1)
                else:
                    output_pred = output_pred_stage_1

                sample_metric = []
                output_pred = output_pred.to(device)

                # Get the base filename without extension
                base_name = os.path.splitext(save_name)[0]

                if args.full_save:
                    # Full save mode: save all output images
                    for i in range(len(outputs_fields)):
                        output_pred_gc = output_pred[i:i+1].clone()
                        resize_size = original_size if args.restore_original_size else None
                        save_image(f"{save_to_dir}/{save_name}_{outputs_fields[i]}.png",
                                   output_pred_gc, resize_size)

                    composite_img = batch["imgs"][inputs_fields[0]]
                    composite_img = torch.clamp(composite_img, 0, 1).to(device)

                    resize_size = original_size if args.restore_original_size else None
                    save_image(f"{save_to_dir}/{save_name}_composite_img.png",
                               composite_img, resize_size)
                else:
                    # Default mode: save only the first/main corrected image with the same name as input
                    if len(output_pred) > 0:
                        output_pred_gc = output_pred[0:1].clone()  # Take the first output as the main corrected image
                        resize_size = original_size if args.restore_original_size else None
                        # Use the same name as input (preserve original extension or use .png)
                        input_ext = os.path.splitext(save_name)[1] or '.png'
                        output_filename = f"{base_name}{input_ext}"
                        save_image(f"{save_to_dir}/{output_filename}",
                                   output_pred_gc, resize_size)

                # Clear GPU cache after each batch to prevent memory accumulation
                if MEMORY_UTILS_AVAILABLE:
                    clear_gpu_memory()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
