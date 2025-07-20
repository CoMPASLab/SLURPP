# this code is modified from Marigold: https://github.com/prs-eth/Marigold

import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from slurpp import load_stage1
from src.util.seeding import seed_all

from src.util.config_util import (
    recursive_load_config,
)

from slurpp.io import save_image, normalize_imgs

from torch.utils.data import DataLoader


from torchvision import transforms
from datasets.UR_revised_dataloader import UnderwaterDataset


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

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    cfg = args.config
    cfg = recursive_load_config(args.config)
    checkpoint_path = args.checkpoint

    output_dir = args.output_dir

    denoise_steps = args.denoise_steps

    seed = args.seed


    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    image_size = getattr(cfg.dataloader, 'image_size', 256)
    image_size = 512
    torch.manual_seed(812)
    np.random.seed(1231)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    val_ds = UnderwaterDataset(image_size= image_size)
    dataloader = DataLoader(val_ds, 
                                num_workers=1, 
                                batch_size=1, 
                                shuffle=False)


    # -------------------- Model --------------------
    dtype = torch.float32

    base_ckpt_dir = os.environ["BASE_CKPT_DIR"]
    model_path = f"{base_ckpt_dir}/stable-diffusion-2"

    print(f"LOADING STAGE 1")

    pipe, inputs_fields, outputs_fields, dual = load_stage1(model_path, checkpoint_path, cfg)
    
    print(f"inputs_fields {inputs_fields}")
    print(f"outputs_fields {outputs_fields}")

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)
    # -------------------- Inference and saving --------------------
    offset =0

    num_inference_steps = 20
    one_step = getattr(cfg, 'one_step', False)
    if one_step:
        print(f"using one step inference")
        num_inference_steps = 1
        pipe.scheduler.config.timestep_spacing = 'trailing'

    for epoch in range(5):
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(dataloader, disable=None)):
                # for k, v in batch["imgs"].items():
                #     print(k)
                #     print(v.shape)
                count = batch_id  + epoch * len(dataloader) + offset
                save_to_dir = f"{output_dir}/train/{count}"

                os.makedirs(save_to_dir, exist_ok=True)
                
                inputs  = []
                for field in inputs_fields:
                    inputs.append(batch["imgs"][field])

                for i in range(len(inputs)):
                    inputs[i] = normalize_imgs(inputs[i])
                
                output_pred = pipe(
                    inputs,
                    denoising_steps=num_inference_steps,
                    show_progress_bar=True,
                    return_latent = True,
                    is_dual = dual,
                )

                
                output_pred_latent = output_pred[1]
                output_pred_stage_1 = output_pred[0]

                
                output_pred = output_pred_stage_1
                # np.save(f"{save_to_dir}/output_pred_latent.npy", output_pred_latent.cpu().numpy()[0])

                sample_metric = []
                output_pred = output_pred.to(device)
                
                # Gamma correction 
                for i in range(len(outputs_fields)):
                    if i == 0:
                        output = torch.clamp(batch['imgs'][outputs_fields[i]], 0, 1).to(device)
                        output_gc = output
                        
                        output_pred_gc = output_pred[i:i+1].clone()
                        os.makedirs(f"{save_to_dir}/{outputs_fields[i]}", exist_ok=True)
                        save_image(f"{save_to_dir}/{outputs_fields[i]}/output_pred_gc.png", output_pred_gc)
                        save_image(f"{save_to_dir}/{outputs_fields[i]}/output_gc.png", output_gc)

                composite_img = batch["imgs"][inputs_fields[0]]
                composite_img = torch.clamp(composite_img, 0, 1).to(device)

                save_image(f"{save_to_dir}/composite_img.png", composite_img)
                