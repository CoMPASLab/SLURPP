# this code is modified from Marigold: https://github.com/prs-eth/Marigold

import logging
import os

import torch
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from .slurpp_pipeline import SlurppPipeline

from diffusers import DDIMScheduler

# Try to import safe loading function
try:
    import sys
    sys.path.append('../..')
    from memory_utils import safe_torch_load, ensure_consistent_dtype
    SAFE_LOAD_AVAILABLE = True
except ImportError:
    SAFE_LOAD_AVAILABLE = False


def safe_load_checkpoint(path):
    """Safely load checkpoint with fallback."""
    if SAFE_LOAD_AVAILABLE:
        return safe_torch_load(path, map_location='cpu')
    else:
        return torch.load(path, map_location='cpu')


def _replace_unet_conv_out(unet, output_imgs=1):
    out_channels = output_imgs * 4
    if out_channels == unet.config["out_channels"]:
        return unet
    _weight = unet.conv_out.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_out.bias.clone()  # [320]
    _bias = _bias.repeat((output_imgs,))  # Keep selected channel(s)
    _weight = _weight.repeat((output_imgs, 1, 1, 1))  # Keep selected channel(s)

    # new conv_in channel
    _n_convout_in_channel = unet.conv_out.in_channels
    _new_conv_out = Conv2d(
        _n_convout_in_channel, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_out.weight = Parameter(_weight)
    _new_conv_out.bias = Parameter(_bias)
    unet.conv_out = _new_conv_out
    logging.info("Unet conv_out layer is replaced")
    # replace config
    unet.config["out_channels"] = out_channels
    logging.info("Unet out config is updated")
    return unet


def _replace_unet_conv_in(unet, input_imgs=1, output_imgs=1):
    in_imgs = input_imgs + output_imgs
    in_channels = in_imgs * 4
    if in_channels == unet.config["in_channels"]:
        return unet
    _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()  # [320]
    in_channels_ori = unet.conv_in.in_channels
    if in_channels_ori == 4:
        _weight = _weight.repeat((1, in_imgs, 1, 1))  # Keep selected channel(s)
        _weight *= (1/in_imgs)
    else:
        _weight_input = _weight[:, :4].repeat((1, input_imgs, 1, 1))  # Keep selected channel(s)
        _weight_input = _weight_input * (1/input_imgs)
        _weight_output = _weight[:, 4:8].repeat((1, output_imgs, 1, 1))  # Keep selected channel(s)
        _weight_output = _weight_output * (1/output_imgs)
        _weight = torch.cat((_weight_input, _weight_output), dim=1)
    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(
        in_channels, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    )
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    logging.info("Unet conv_in layer is replaced")
    # replace config
    unet.config["in_channels"] = in_channels
    logging.info("Unet config is updated")
    return unet


def load_stage1(model_path, checkpoint_path, cfg, low_memory=False):
    # Try selective loading for low memory mode
    if low_memory and SAFE_LOAD_AVAILABLE:
        try:
            from memory_utils import load_pipeline_selective
            components = load_pipeline_selective(model_path, low_memory=True)
            if components:
                # Create pipeline with components on CPU
                pipe = SlurppPipeline(
                    unet=components['unet'],
                    vae=components['vae'],
                    scheduler=components['scheduler'],
                    text_encoder=components['text_encoder'],
                    tokenizer=components['tokenizer'],
                )
                print("✓ Pipeline created with selective loading")
            else:
                raise Exception("Selective loading failed")
        except Exception as e:
            print(f"⚠ Selective loading failed: {e}, falling back to standard loading")
            # Fallback to standard loading
            pipe = SlurppPipeline.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
            )
    else:
        # Standard loading
        pipe = SlurppPipeline.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
        )

    inputs_fields = getattr(cfg.trainer, 'inputs', ['diff'])
    outputs_fields = getattr(cfg.trainer, 'output', ['bc', 'ill'])
    dual = getattr(cfg, 'dual', False)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if dual:
        from my_diffusers.dual_unet_condition import DualUNetCondition
        print("using dual unet")
        pipe.unet = DualUNetCondition(unet_path1=model_path, unet_path2=model_path)
        pipe.unet.unet1 = _replace_unet_conv_in(pipe.unet.unet1, len(inputs_fields), len(outputs_fields))
        pipe.unet.unet1 = _replace_unet_conv_out(pipe.unet.unet1, len(outputs_fields))
        inputs_fields2 = getattr(cfg.trainer, 'inputs2', ['diff'])
        outputs_fields2 = getattr(cfg.trainer, 'output2', ['bc', 'ill'])
        pipe.unet.unet2 = _replace_unet_conv_in(pipe.unet.unet2, len(inputs_fields2), len(outputs_fields2))
        pipe.unet.unet2 = _replace_unet_conv_out(pipe.unet.unet2, len(outputs_fields2))
        pipe.unet.add_additional_params()
        # Load with safe loading method to handle compatibility issues
        unet1_path = os.path.join(checkpoint_path, 'unet1', 'diffusion_pytorch_model.bin')
        checkpoint1 = safe_load_checkpoint(unet1_path)
        pipe.unet.unet1.load_state_dict(checkpoint1)
        del checkpoint1  # Free memory immediately

        unet2_path = os.path.join(checkpoint_path, 'unet2', 'diffusion_pytorch_model.bin')
        checkpoint2 = safe_load_checkpoint(unet2_path)
        pipe.unet.unet2.load_state_dict(checkpoint2)
        del checkpoint2  # Free memory immediately
        inputs_fields = inputs_fields + inputs_fields2
        outputs_fields = outputs_fields + outputs_fields2
    else:
        pipe.unet = _replace_unet_conv_in(pipe.unet, len(inputs_fields), len(outputs_fields))
        pipe.unet = _replace_unet_conv_out(pipe.unet, len(outputs_fields))
        # Load with safe loading method to handle compatibility issues
        unet_path = os.path.join(checkpoint_path, 'unet', 'diffusion_pytorch_model.bin')
        checkpoint = safe_load_checkpoint(unet_path)
        pipe.unet.load_state_dict(checkpoint)
        del checkpoint  # Free memory immediately

    # Ensure dtype consistency to prevent Half/Float errors
    if SAFE_LOAD_AVAILABLE:
        pipe = ensure_consistent_dtype(pipe, use_half_precision=False)

    return pipe, inputs_fields, outputs_fields, dual
