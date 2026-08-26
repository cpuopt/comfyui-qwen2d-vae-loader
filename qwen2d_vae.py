"""ComfyUI VAE wrapper for the two-dimensional Qwen VAE architecture."""

import logging

import torch

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils

from .qwen2d_arch import Qwen2DVAEModel


def is_qwen2d_state_dict(state_dict):
    """Return whether a state dict has the characteristic Qwen2D VAE keys."""
    if not state_dict:
        return False
    required_keys = (
        "decoder.mid_block.attentions.0.norm.gamma",
        "quant_conv.weight",
        "post_quant_conv.weight",
        "decoder.conv_in.weight",
        "decoder.conv_out.weight",
        "decoder.norm_out.gamma",
    )
    return all(key in state_dict for key in required_keys) and state_dict[
        "decoder.conv_in.weight"
    ].ndim == 4


def _strip_singleton_temporal(tensor):
    if tensor.ndim == 5 and tensor.shape[2] == 1:
        return tensor[:, :, 0], True
    return tensor, False


def _flatten_temporal_batch(tensor):
    if tensor.ndim != 5:
        return tensor, None
    batch, channels, frames, height, width = tensor.shape
    tensor = tensor.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    return tensor, (batch, frames)


def _restore_temporal_batch(tensor, frame_info):
    if frame_info is None:
        return tensor
    batch, frames = frame_info
    channels, height, width = tensor.shape[1:]
    return tensor.reshape(batch, frames, channels, height, width).permute(
        0, 2, 1, 3, 4
    )


class Qwen2DVAE(comfy.sd.VAE):
    """A VAE with ComfyUI's public VAE interface, without patching ComfyUI."""

    def __init__(self, state_dict, device=None, dtype=None):
        if not is_qwen2d_state_dict(state_dict):
            raise ValueError(
                "The selected file is not a supported Qwen2D VAE checkpoint. "
                "Expected the standalone Qwen2D VAE weights."
            )

        self.downscale_ratio = (lambda value: value, 8, 8)
        self.upscale_ratio = (lambda value: value, 8, 8)
        self.downscale_index_formula = (1, 8, 8)
        self.upscale_index_formula = (1, 8, 8)
        self.latent_channels = state_dict["decoder.conv_in.weight"].shape[1]
        self.latent_dim = 3
        self.output_channels = state_dict["decoder.conv_out.weight"].shape[0]
        self.pad_channel_value = None
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: torch.clamp(
            (image + 1.0) / 2.0, min=0.0, max=1.0
        )
        self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        self.disable_offload = False
        self.not_video = True
        self.size = None
        self.extra_1d_channel = None
        self.crop_input = True
        self.audio_sample_rate = 44100
        self.memory_used_encode = lambda shape, value_dtype: (
            1400
            * (shape[2] if len(shape) == 5 else 1)
            * shape[-2]
            * shape[-1]
            * model_management.dtype_size(value_dtype)
        )
        self.memory_used_decode = lambda shape, value_dtype: (
            1800
            * (shape[2] if len(shape) == 5 else 1)
            * shape[-2]
            * shape[-1]
            * 64
            * model_management.dtype_size(value_dtype)
        )

        config = {
            "base_dim": state_dict["decoder.norm_out.gamma"].shape[0],
            "z_dim": self.latent_channels,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "image_channels": self.output_channels,
            "dropout": 0.0,
        }
        self.first_stage_model = Qwen2DVAEModel(**config).eval()

        self.device = device or model_management.vae_device()
        offload_device = model_management.vae_offload_device()
        self.vae_dtype = dtype or model_management.vae_dtype(
            self.device, self.working_dtypes
        )
        self.first_stage_model.to(self.vae_dtype)
        model_management.archive_model_dtypes(self.first_stage_model)
        self.output_device = model_management.intermediate_device()

        patcher_class = getattr(
            comfy.model_patcher,
            "CoreModelPatcher",
            comfy.model_patcher.ModelPatcher,
        )
        if self.disable_offload:
            patcher_class = comfy.model_patcher.ModelPatcher
        self.patcher = patcher_class(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device,
        )

        is_dynamic = getattr(self.patcher, "is_dynamic", lambda: False)()
        missing, unexpected = self.first_stage_model.load_state_dict(
            state_dict, strict=False, assign=is_dynamic
        )
        if missing:
            logging.warning("Missing Qwen2D VAE keys: %s", missing)
        if unexpected:
            logging.debug("Unused Qwen2D VAE keys: %s", unexpected)
        logging.info(
            "Qwen2D VAE load device: %s, offload device: %s, dtype: %s",
            self.device,
            offload_device,
            self.vae_dtype,
        )
        self.model_size()

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(
            samples.shape[3], samples.shape[2], tile_x, tile_y, overlap
        )
        steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(
            samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap
        )
        steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(
            samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap
        )
        progress = comfy.utils.ProgressBar(steps)
        upscale_amount = self.spacial_compression_decode()
        decode_fn = lambda value: self.first_stage_model.decode(
            value.to(self.vae_dtype).to(self.device)
        ).float()
        output = (
            comfy.utils.tiled_scale(
                samples,
                decode_fn,
                tile_x // 2,
                tile_y * 2,
                overlap,
                upscale_amount=upscale_amount,
                output_device=self.output_device,
                pbar=progress,
            )
            + comfy.utils.tiled_scale(
                samples,
                decode_fn,
                tile_x * 2,
                tile_y // 2,
                overlap,
                upscale_amount=upscale_amount,
                output_device=self.output_device,
                pbar=progress,
            )
            + comfy.utils.tiled_scale(
                samples,
                decode_fn,
                tile_x,
                tile_y,
                overlap,
                upscale_amount=upscale_amount,
                output_device=self.output_device,
                pbar=progress,
            )
        ) / 3.0
        return self.process_output(output)

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        steps = pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(
            pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap
        )
        steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(
            pixel_samples.shape[3],
            pixel_samples.shape[2],
            tile_x // 2,
            tile_y * 2,
            overlap,
        )
        steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(
            pixel_samples.shape[3],
            pixel_samples.shape[2],
            tile_x * 2,
            tile_y // 2,
            overlap,
        )
        progress = comfy.utils.ProgressBar(steps)
        upscale_amount = 1 / self.spacial_compression_encode()
        encode_fn = lambda value: self.first_stage_model.encode(
            self.process_input(value).to(self.vae_dtype).to(self.device)
        ).float()
        common = {
            "upscale_amount": upscale_amount,
            "out_channels": self.latent_channels,
            "output_device": self.output_device,
            "pbar": progress,
        }
        samples = comfy.utils.tiled_scale(
            pixel_samples, encode_fn, tile_x, tile_y, overlap, **common
        )
        samples += comfy.utils.tiled_scale(
            pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, **common
        )
        samples += comfy.utils.tiled_scale(
            pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, **common
        )
        return samples / 3.0

    def decode(self, samples_in, vae_options=None):
        self.throw_exception_if_invalid()
        vae_options = vae_options or {}
        samples_in, squeezed_temporal = _strip_singleton_temporal(samples_in)
        samples_in, frame_info = _flatten_temporal_batch(samples_in)
        pixel_samples = None
        try:
            memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
            model_management.load_models_gpu(
                [self.patcher],
                memory_required=memory_used,
                force_full_load=self.disable_offload,
            )
            free_memory = self.patcher.get_free_memory(self.device)
            batch_size = max(1, int(free_memory / max(1, memory_used)))
            for start in range(0, samples_in.shape[0], batch_size):
                samples = samples_in[start : start + batch_size].to(
                    device=self.device, dtype=self.vae_dtype
                )
                output = self.process_output(
                    self.first_stage_model.decode(samples, **vae_options)
                    .to(self.output_device)
                    .float()
                )
                if pixel_samples is None:
                    pixel_samples = torch.empty(
                        (samples_in.shape[0],) + tuple(output.shape[1:]),
                        device=self.output_device,
                    )
                pixel_samples[start : start + batch_size] = output
        except model_management.OOM_EXCEPTION:
            logging.warning(
                "Out of memory during Qwen2D VAE decoding; retrying tiled."
            )
            pixel_samples = self.decode_tiled_(samples_in)

        pixel_samples = _restore_temporal_batch(pixel_samples, frame_info)
        if (
            squeezed_temporal
            and pixel_samples.ndim == 5
            and pixel_samples.shape[2] == 1
        ):
            pixel_samples = pixel_samples[:, :, 0]
        return pixel_samples.to(self.output_device).movedim(1, -1)

    def decode_tiled(
        self,
        samples,
        tile_x=None,
        tile_y=None,
        overlap=None,
        tile_t=None,
        overlap_t=None,
    ):
        self.throw_exception_if_invalid()
        samples, squeezed_temporal = _strip_singleton_temporal(samples)
        samples, frame_info = _flatten_temporal_batch(samples)
        memory_used = self.memory_used_decode(samples.shape, self.vae_dtype)
        model_management.load_models_gpu(
            [self.patcher],
            memory_required=memory_used,
            force_full_load=self.disable_offload,
        )
        options = {}
        if tile_x is not None:
            options["tile_x"] = tile_x
        if tile_y is not None:
            options["tile_y"] = tile_y
        if overlap is not None:
            options["overlap"] = overlap
        output = self.decode_tiled_(samples, **options)
        output = _restore_temporal_batch(output, frame_info)
        if squeezed_temporal and output.ndim == 5 and output.shape[2] == 1:
            output = output[:, :, 0]
        return output.movedim(1, -1)

    def encode(self, pixel_samples):
        self.throw_exception_if_invalid()
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples).movedim(-1, 1)
        if pixel_samples.ndim == 4:
            pixel_samples = pixel_samples.unsqueeze(2)
        pixel_samples, squeezed_temporal = _strip_singleton_temporal(pixel_samples)
        pixel_samples, frame_info = _flatten_temporal_batch(pixel_samples)
        samples = None
        try:
            memory_used = self.memory_used_encode(
                pixel_samples.shape, self.vae_dtype
            )
            model_management.load_models_gpu(
                [self.patcher],
                memory_required=memory_used,
                force_full_load=self.disable_offload,
            )
            free_memory = self.patcher.get_free_memory(self.device)
            batch_size = max(1, int(free_memory / max(1, memory_used)))
            for start in range(0, pixel_samples.shape[0], batch_size):
                pixels = self.process_input(
                    pixel_samples[start : start + batch_size]
                ).to(device=self.device, dtype=self.vae_dtype)
                output = (
                    self.first_stage_model.encode(pixels)
                    .to(self.output_device)
                    .float()
                )
                if samples is None:
                    samples = torch.empty(
                        (pixel_samples.shape[0],) + tuple(output.shape[1:]),
                        device=self.output_device,
                    )
                samples[start : start + batch_size] = output
        except model_management.OOM_EXCEPTION:
            logging.warning(
                "Out of memory during Qwen2D VAE encoding; retrying tiled."
            )
            samples = self.encode_tiled_(pixel_samples)

        samples = _restore_temporal_batch(samples, frame_info)
        if squeezed_temporal:
            samples = samples.unsqueeze(2)
        return samples

    def encode_tiled(
        self,
        pixel_samples,
        tile_x=None,
        tile_y=None,
        overlap=None,
        tile_t=None,
        overlap_t=None,
    ):
        self.throw_exception_if_invalid()
        pixel_samples = self.vae_encode_crop_pixels(pixel_samples).movedim(-1, 1)
        if pixel_samples.ndim == 4:
            pixel_samples = pixel_samples.unsqueeze(2)
        pixel_samples, squeezed_temporal = _strip_singleton_temporal(pixel_samples)
        pixel_samples, frame_info = _flatten_temporal_batch(pixel_samples)
        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
        model_management.load_models_gpu(
            [self.patcher],
            memory_required=memory_used,
            force_full_load=self.disable_offload,
        )
        options = {}
        if tile_x is not None:
            options["tile_x"] = tile_x
        if tile_y is not None:
            options["tile_y"] = tile_y
        if overlap is not None:
            options["overlap"] = overlap
        samples = self.encode_tiled_(pixel_samples, **options)
        samples = _restore_temporal_batch(samples, frame_info)
        if squeezed_temporal:
            samples = samples.unsqueeze(2)
        return samples


__all__ = ["Qwen2DVAE", "is_qwen2d_state_dict"]
