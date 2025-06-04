# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import torch
import logging
import collections

import nodes
import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths

from .ops import GGMLOps, move_patch_to_device
from .loader import gguf_sd_loader, gguf_clip_loader
from .dequant import is_quantized, is_torch_compatible

def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    mmap_released = False
    def load(self, *args, force_patch_weights=False, **kwargs):
        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        if src_cls != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n

class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "bootleg"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None, node_id=None):
        import os
        import time
        
        # Flag: 2025-06-04 16:55 - Added retry logic for model loading
        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)
            
        # Force refresh the cache to ensure we see the latest files
        if "unet_gguf" in folder_paths.filename_list_cache:
            del folder_paths.filename_list_cache["unet_gguf"]
        
        # Try to find and load the model with retries
        max_attempts = 5
        attempt = 0
        unet_path = None
        
        while attempt < max_attempts:
            try:
                unet_path = folder_paths.get_full_path("unet_gguf", unet_name)
                if unet_path and os.path.exists(unet_path):
                    break
            except Exception as e:
                pass
                
            # If not found, wait a bit and try again (in case it's still being written)
            time.sleep(1)
            
            # Refresh the cache again
            if "unet_gguf" in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache["unet_gguf"]
            folder_paths.get_filename_list("unet_gguf")
            
            attempt += 1
        
        if not unet_path or not os.path.exists(unet_path):
            raise ValueError(f"GGUF model {unet_name} not found after {max_attempts} attempts")
        
        # Load the model
        sd = gguf_sd_loader(unet_path)
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)

class UnetLoaderGGUFAdvanced(UnetLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": ("STRING", {"multiline": False, "default": ""}),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }
    TITLE = "Unet Loader (GGUF/Advanced)"

class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name": ("STRING", {"multiline": False, "default": ""}),
                "type": base["required"]["type"],
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "bootleg"
    TITLE = "CLIPLoader (GGUF)"

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    def load_data(self, ckpt_paths):
        clip_data = []
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                sd = gguf_clip_loader(p)
            else:
                sd = comfy.utils.load_torch_file(p, safe_load=True)
                if "scaled_fp8" in sd: # NOTE: Scaled FP8 would require different custom ops, but only one can be active
                    raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported! Use regular CLIP loader or switch model(s)\n({p})")
            clip_data.append(sd)
        return clip_data

    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type = clip_type,
            state_dicts = clip_data,
            model_options = {
                "custom_operations": GGMLOps,
                "initial_device": comfy.model_management.text_encoder_offload_device()
            },
            embedding_directory = folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        return clip

    def load_clip(self, clip_name, type="stable_diffusion", node_id=None):
        import os
        import time
        
        # Flag: 2025-06-04 16:58 - Added retry logic for model loading
        
        # Force refresh the cache to ensure we see the latest files
        if "clip_gguf" in folder_paths.filename_list_cache:
            del folder_paths.filename_list_cache["clip_gguf"]
        
        # Try to find and load the model with retries
        max_attempts = 5
        attempt = 0
        clip_path = None
        
        while attempt < max_attempts:
            try:
                clip_path = folder_paths.get_full_path("clip_gguf", clip_name)
                if clip_path and os.path.exists(clip_path):
                    break
                    
                # Try regular clip folder if not found in clip_gguf
                clip_path = folder_paths.get_full_path("clip", clip_name)
                if clip_path and os.path.exists(clip_path):
                    break
            except Exception as e:
                pass
                
            # If not found, wait a bit and try again (in case it's still being written)
            time.sleep(1)
            
            # Refresh the cache again
            if "clip_gguf" in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache["clip_gguf"]
            if "clip" in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache["clip"]
                
            folder_paths.get_filename_list("clip_gguf")
            folder_paths.get_filename_list("clip")
            
            attempt += 1
        
        if not clip_path or not os.path.exists(clip_path):
            raise ValueError(f"CLIP model {clip_name} not found after {max_attempts} attempts")
            
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher([clip_path], clip_type, self.load_data([clip_path])),)

class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name1": ("STRING", {"multiline": False, "default": ""}),
                "clip_name2": ("STRING", {"multiline": False, "default": ""}),
                "type": base["required"]["type"],
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    TITLE = "DualCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, type, node_id=None):
        import os
        import time
        
        # Flag: 2025-06-04 17:00 - Added retry logic for model loading
        
        # Force refresh the cache
        for cache_key in ["clip_gguf", "clip"]:
            if cache_key in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache[cache_key]
        
        # Helper function to find clip path with retries
        def find_clip_path(name, max_attempts=5):
            for attempt in range(max_attempts):
                try:
                    # Try clip_gguf folder first
                    path = folder_paths.get_full_path("clip_gguf", name)
                    if path and os.path.exists(path):
                        return path
                        
                    # Try regular clip folder if not found in clip_gguf
                    path = folder_paths.get_full_path("clip", name)
                    if path and os.path.exists(path):
                        return path
                except Exception as e:
                    pass
                    
                # If not found, wait a bit and try again
                time.sleep(1)
                
                # Refresh the cache again
                for cache_key in ["clip_gguf", "clip"]:
                    if cache_key in folder_paths.filename_list_cache:
                        del folder_paths.filename_list_cache[cache_key]
                    folder_paths.get_filename_list(cache_key)
            
            # If we get here, we couldn't find the file
            raise ValueError(f"CLIP model {name} not found after {max_attempts} attempts")
        
        # Find paths for both models
        clip_path1 = find_clip_path(clip_name1)
        clip_path2 = find_clip_path(clip_name2)
        
        clip_paths = (clip_path1, clip_path2)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)

class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": ("STRING", {"multiline": False, "default": ""}),
                "clip_name2": ("STRING", {"multiline": False, "default": ""}),
                "clip_name3": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    TITLE = "TripleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3", node_id=None):
        import os
        import time
        
        # Flag: 2025-06-04 17:00 - Added retry logic for model loading
        
        # Force refresh the cache
        for cache_key in ["clip_gguf", "clip"]:
            if cache_key in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache[cache_key]
        
        # Helper function to find clip path with retries
        def find_clip_path(name, max_attempts=5):
            for attempt in range(max_attempts):
                try:
                    # Try clip_gguf folder first
                    path = folder_paths.get_full_path("clip_gguf", name)
                    if path and os.path.exists(path):
                        return path
                        
                    # Try regular clip folder if not found in clip_gguf
                    path = folder_paths.get_full_path("clip", name)
                    if path and os.path.exists(path):
                        return path
                except Exception as e:
                    pass
                    
                # If not found, wait a bit and try again
                time.sleep(1)
                
                # Refresh the cache again
                for cache_key in ["clip_gguf", "clip"]:
                    if cache_key in folder_paths.filename_list_cache:
                        del folder_paths.filename_list_cache[cache_key]
                    folder_paths.get_filename_list(cache_key)
            
            # If we get here, we couldn't find the file
            raise ValueError(f"CLIP model {name} not found after {max_attempts} attempts")
        
        # Find paths for all models
        clip_path1 = find_clip_path(clip_name1)
        clip_path2 = find_clip_path(clip_name2)
        clip_path3 = find_clip_path(clip_name3)
        
        clip_paths = (clip_path1, clip_path2, clip_path3)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)

class QuadrupleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "clip_name1": ("STRING", {"multiline": False, "default": ""}),
            "clip_name2": ("STRING", {"multiline": False, "default": ""}),
            "clip_name3": ("STRING", {"multiline": False, "default": ""}),
            "clip_name4": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    TITLE = "QuadrupleCLIPLoader (GGUF)"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion", node_id=None):
        import os
        import time
        
        # Flag: 2025-06-04 17:00 - Added retry logic for model loading
        
        # Force refresh the cache
        for cache_key in ["clip_gguf", "clip"]:
            if cache_key in folder_paths.filename_list_cache:
                del folder_paths.filename_list_cache[cache_key]
        
        # Helper function to find clip path with retries
        def find_clip_path(name, max_attempts=5):
            for attempt in range(max_attempts):
                try:
                    # Try clip_gguf folder first
                    path = folder_paths.get_full_path("clip_gguf", name)
                    if path and os.path.exists(path):
                        return path
                        
                    # Try regular clip folder if not found in clip_gguf
                    path = folder_paths.get_full_path("clip", name)
                    if path and os.path.exists(path):
                        return path
                except Exception as e:
                    pass
                    
                # If not found, wait a bit and try again
                time.sleep(1)
                
                # Refresh the cache again
                for cache_key in ["clip_gguf", "clip"]:
                    if cache_key in folder_paths.filename_list_cache:
                        del folder_paths.filename_list_cache[cache_key]
                    folder_paths.get_filename_list(cache_key)
            
            # If we get here, we couldn't find the file
            raise ValueError(f"CLIP model {name} not found after {max_attempts} attempts")
        
        # Find paths for all models
        clip_path1 = find_clip_path(clip_name1)
        clip_path2 = find_clip_path(clip_name2)
        clip_path3 = find_clip_path(clip_name3)
        clip_path4 = find_clip_path(clip_name4)
        
        clip_paths = (clip_path1, clip_path2, clip_path3, clip_path4)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)

NODE_CLASS_MAPPINGS = {
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "CLIPLoaderGGUF": CLIPLoaderGGUF,
    "DualCLIPLoaderGGUF": DualCLIPLoaderGGUF,
    "TripleCLIPLoaderGGUF": TripleCLIPLoaderGGUF,
    "QuadrupleCLIPLoaderGGUF": QuadrupleCLIPLoaderGGUF,
    "UnetLoaderGGUFAdvanced": UnetLoaderGGUFAdvanced,
}
