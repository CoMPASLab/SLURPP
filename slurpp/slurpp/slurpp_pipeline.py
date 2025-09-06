# this code is modified from Marigold: https://github.com/prs-eth/Marigold
from typing import Optional, Union
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel
)
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


class SlurppPipeline(DiffusionPipeline):

    rgb_latent_scale_factor = 0.18215
    pred_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None
        self.skip_connection = False
        self.composite_latents = None
        self.pass_zero_conv = False
        self.vae_cld = None

    @torch.no_grad()
    def __call__(
        self,
        inputs,
        denoising_steps: Optional[int] = None,
        processing_res: Optional[int] = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
        multi_patch: bool = False,
        window_size: int = 96,
        stride: int = 48,
        is_dual: bool = False,
        return_latent: bool = False,
    ):
        pred = self.single_infer(
            rgb_in=inputs,
            num_inference_steps=denoising_steps,
            show_pbar=show_progress_bar,
            generator=generator,
            is_dual=is_dual,
            return_latent=return_latent
        )

        return pred

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt with dynamic device handling
        """
        # Skip text encoding if text encoder is disabled
        if self.text_encoder is None:
            # Create a dummy embedding with correct shape and dtype
            # Text embeddings should be [1, 77, 1024] for this model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.empty_text_embed = torch.zeros(1, 77, 1024, dtype=self.dtype, device=device)
            return

        # Also check if tokenizer is available
        if self.tokenizer is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.empty_text_embed = torch.zeros(1, 77, 1024, dtype=self.dtype, device=device)
            return

        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Handle dynamic text encoder loading
        text_encoder_was_on_cpu = False
        target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Check if text encoder is on CPU and needs to be moved
        if hasattr(self.text_encoder, 'device') and str(self.text_encoder.device) == 'cpu':
            text_encoder_was_on_cpu = True
            self.text_encoder = self.text_encoder.to(target_device)

        # Move input IDs to the same device as text encoder with correct dtype
        # Input IDs must be Long/Int tensors for embedding layers
        text_input_ids = text_inputs.input_ids.to(
            device=self.text_encoder.device, dtype=torch.long
        )

        # Encode text with explicit position_ids to avoid dtype issues
        # Create position_ids explicitly as Long tensors
        seq_length = text_input_ids.shape[-1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=text_input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(text_input_ids)

        # Use the text model directly with explicit position_ids
        self.empty_text_embed = self.text_encoder.text_model(
            input_ids=text_input_ids,
            position_ids=position_ids
        )[0].to(self.dtype)

        # Move text encoder back to CPU if it was there originally
        if text_encoder_was_on_cpu:
            self.text_encoder = self.text_encoder.to('cpu')

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        is_dual: bool = False,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """
        Perform an individual prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted image.
        """
        device = rgb_in[0].device
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]
        # rgb_in = rgb_in.to(device)
        batch_size = 1

        # Move inputs to device one by one to save memory
        rgb_latents = []
        for rgb in rgb_in:
            rgb_device = rgb.to(device, dtype=self.dtype)
            rgb_latents.append(self.encode_rgb(rgb_device))  # [B, 4, h, w]
            # Clear intermediate tensor from memory
            del rgb_device
        rgb_latents = torch.cat(rgb_latents, dim=1)  # [B, 4*len(rgb_in), h, w]
        # Ensure rgb_latents is on the correct device
        rgb_latents = rgb_latents.to(device)

        out_shape = list(rgb_latents[:, :4].shape)
        if not is_dual:
            out_shape[1] = self.unet.config["out_channels"]  # [B, 4, h, w]
        else:
            # [B, 8, h, w]
            out_shape[1] = (self.unet.unet1.config["out_channels"] +
                            self.unet.unet2.config["out_channels"])
        if num_inference_steps == 1:
            pred_latent = torch.zeros(out_shape,
                                      device=device,
                                      dtype=self.dtype)  # [B, 4, h, w]
        else:
            pred_latent = torch.randn(out_shape,
                                      device=device,
                                      dtype=self.dtype,
                                      generator=generator)  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (batch_size, 1, 1)
        ).to(device, dtype=self.dtype)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for _, t in iterable:
            # Ensure all tensors are on the same device before concatenation
            rgb_latents = rgb_latents.to(device)
            pred_latent = pred_latent.to(device)

            unet_input = torch.cat(
                [rgb_latents, pred_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            pred_latent = self.scheduler.step(
                noise_pred, t, pred_latent, generator=generator
            ).prev_sample

        if return_latent:
            return (self.decode_images(pred_latent), pred_latent)

        return self.decode_images(pred_latent)

    def encode_rgb(self, rgb_in: torch.Tensor, skip_connection: bool = None) -> torch.Tensor:
        """
        Encode RGB image into latent with dynamic VAE loading.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # Move VAE to GPU temporarily if it's on CPU
        vae_was_on_cpu = False
        if hasattr(self.vae, 'device') and str(self.vae.device) == 'cpu':
            vae_was_on_cpu = True
            # Ensure all parameters and buffers are moved to GPU
            self.vae = self.vae.to(rgb_in.device)
            # Force consistency of all submodules
            for module in self.vae.modules():
                module.to(rgb_in.device)

        # encode
        if skip_connection is None:
            skip_connection = self.skip_connection
        if skip_connection and self.vae_cld is not None:
            # Clear GPU cache before heavy VAE operation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Move VAE CLD to GPU if needed
            vae_cld_was_on_cpu = False
            if hasattr(self.vae_cld, 'device') and str(self.vae_cld.device) == 'cpu':
                vae_cld_was_on_cpu = True
                # Ensure all parameters and buffers are moved to GPU
                self.vae_cld = self.vae_cld.to(rgb_in.device)
                # Force consistency of all submodules
                for module in self.vae_cld.modules():
                    module.to(rgb_in.device)

            # Ensure input tensor matches the model's device and dtype exactly
            target_device = next(self.vae_cld.parameters()).device
            target_dtype = next(self.vae_cld.parameters()).dtype
            rgb_in = rgb_in.to(device=target_device, dtype=target_dtype)

            # Use gradient checkpointing for memory efficiency
            with torch.cuda.amp.autocast(enabled=False):
                out = self.vae_cld.encode(rgb_in, skip_connection=True)

            self.composite_latents = out.composite_latents
            self.pass_zero_conv = False
            latents = out.latent_dist.sample()

            # Move VAE CLD back to CPU immediately and clear cache
            if vae_cld_was_on_cpu:
                self.vae_cld = self.vae_cld.to('cpu')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Ensure input tensor matches the model's device and dtype exactly
            target_device = next(self.vae.parameters()).device
            target_dtype = next(self.vae.parameters()).dtype
            rgb_in = rgb_in.to(device=target_device, dtype=target_dtype)

            out = self.vae.encode(rgb_in)
            latents = out.latent_dist.sample()

        # Move VAE back to CPU
        if vae_was_on_cpu:
            self.vae = self.vae.to('cpu')

        # scale latent
        rgb_latent = latents * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_images(self, img_latent: torch.Tensor) -> torch.Tensor:
        # Move VAE to GPU temporarily if it's on CPU
        vae_was_on_cpu = False
        if hasattr(self.vae, 'device') and str(self.vae.device) == 'cpu':
            vae_was_on_cpu = True
            # Ensure all parameters and buffers are moved to GPU
            self.vae = self.vae.to(img_latent.device)
            # Force consistency of all submodules
            for module in self.vae.modules():
                module.to(img_latent.device)

        img_latent = img_latent / self.rgb_latent_scale_factor
        if self.skip_connection and self.vae_cld is not None:
            # Move VAE CLD to GPU if needed
            vae_cld_was_on_cpu = False
            if hasattr(self.vae_cld, 'device') and str(self.vae_cld.device) == 'cpu':
                vae_cld_was_on_cpu = True
                # Ensure all parameters and buffers are moved to GPU
                self.vae_cld = self.vae_cld.to(img_latent.device)
                # Force consistency of all submodules
                for module in self.vae_cld.modules():
                    module.to(img_latent.device)

            if not self.pass_zero_conv:
                self.composite_latents = self.vae_cld.pass_zero_conv(self.composite_latents)
                self.pass_zero_conv = True

            # Ensure img_latent matches VAE CLD's device and dtype
            target_device = next(self.vae_cld.parameters()).device
            target_dtype = next(self.vae_cld.parameters()).dtype
            img_latent = img_latent.to(device=target_device, dtype=target_dtype)

            if img_latent.shape[1] == 4:
                image = self.vae_cld.decode(img_latent, composite_latents=self.composite_latents).sample
            else:
                img_splits = torch.split(img_latent, 4, dim=1)
                decoded_images = []

                # cld decode for the first split(clear)
                decoded_img = self.vae_cld.decode(
                    img_splits[0], composite_latents=self.composite_latents).sample
                decoded_images.append(decoded_img)

                # normal decode for the rest
                remaining_splits = torch.cat(img_splits[1:], dim=0)
                # Ensure remaining splits match VAE's device and dtype
                vae_device = next(self.vae.parameters()).device
                vae_dtype = next(self.vae.parameters()).dtype
                remaining_splits = remaining_splits.to(device=vae_device, dtype=vae_dtype)

                decoded_images.append(self.vae.decode(remaining_splits).sample)

                # Ensure all decoded images are on the same device before concatenation
                target_device = decoded_images[0].device
                for i in range(len(decoded_images)):
                    decoded_images[i] = decoded_images[i].to(target_device)

                image = torch.cat(decoded_images, dim=0)

            # Move VAE CLD back to CPU
            if vae_cld_was_on_cpu:
                self.vae_cld = self.vae_cld.to('cpu')

        else:
            # Ensure img_latent matches VAE's device and dtype
            target_device = next(self.vae.parameters()).device
            target_dtype = next(self.vae.parameters()).dtype
            img_latent = img_latent.to(device=target_device, dtype=target_dtype)

            if img_latent.shape[1] == 4:
                image = self.vae.decode(img_latent).sample
            else:
                img_splits = img_latent.reshape(
                    img_latent.shape[0] * (img_latent.shape[1] // 4), 4,
                    img_latent.shape[2], img_latent.shape[3]
                )
                # Ensure img_splits matches VAE's device and dtype
                vae_device = next(self.vae.parameters()).device
                vae_dtype = next(self.vae.parameters()).dtype
                img_splits = img_splits.to(device=vae_device, dtype=vae_dtype)

                image = self.vae.decode(img_splits).sample

        # Move VAE back to CPU
        if vae_was_on_cpu:
            self.vae = self.vae.to('cpu')

        image = (image / 2 + 0.5).clamp(0, 1)
        return image  # End of pipeline
