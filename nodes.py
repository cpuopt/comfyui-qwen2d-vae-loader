"""ComfyUI node definitions."""

import comfy.utils
import folder_paths

from .qwen2d_vae import Qwen2DVAE


class Qwen2DVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"),),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Loads a standalone Qwen2D VAE without changing ComfyUI's built-in "
        "VAE Loader. Place the checkpoint in models/vae."
    )

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        state_dict = comfy.utils.load_torch_file(vae_path, safe_load=True)
        return (Qwen2DVAE(state_dict),)


NODE_CLASS_MAPPINGS = {
    "Qwen2DVAELoader": Qwen2DVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2DVAELoader": "Qwen2D VAE Loader",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
