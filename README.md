# comfyui-qwen2d-vae-loader

Adds a standalone **Qwen2D VAE Loader** node to ComfyUI. It loads the
[Qwen2D-VAE](https://huggingface.co/Anzhc/Qwen2D-VAE) architecture without
patching or replacing ComfyUI's built-in `VAE Loader`.

## Installation

Clone this repository into `ComfyUI/custom_nodes`, then restart ComfyUI.

Put the Qwen2D VAE checkpoint in `ComfyUI/models/vae`. Add **Qwen2D VAE
Loader** from the `loaders` category and select the checkpoint. Its `VAE`
output can be connected to the standard VAE Encode/Decode nodes.

The loader validates the selected checkpoint and reports an error when it is
not a supported Qwen2D VAE.
