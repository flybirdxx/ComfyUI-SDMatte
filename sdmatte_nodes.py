import os
import torch
from torchvision import transforms

import folder_paths
import comfy

# Get the models directory from ComfyUI
MODEL_DIR = os.path.join(folder_paths.models_dir, "SDMatte")

# Register the SDMatte folder path with ComfyUI
folder_paths.add_model_folder_path("SDMatte", MODEL_DIR)

MODEL_URLS = {
    "SDMatte.safetensors": "https://huggingface.co/1038lab/SDMatte/resolve/main/SDMatte.safetensors",
    "SDMatte_plus.safetensors": "https://huggingface.co/1038lab/SDMatte/resolve/main/SDMatte_plus.safetensors"
}

# Files to fetch from Manojb/stable-diffusion-2-1-base (configs only)
SD21_MANOJB_FILES = {
    "model_index.json": "model_index.json",
    "text_encoder/config.json": "text_encoder/config.json",
    "vae/config.json": "vae/config.json",
    "unet/config.json": "unet/config.json",
    "scheduler/scheduler_config.json": "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json": "tokenizer/tokenizer_config.json",
    "tokenizer/merges.txt": "tokenizer/merges.txt",
    "tokenizer/vocab.json": "tokenizer/vocab.json",
    "tokenizer/special_tokens_map.json": "tokenizer/special_tokens_map.json",
    "feature_extractor/preprocessor_config.json": "feature_extractor/preprocessor_config.json",
}


def ensure_sd21_from_manojb(sd21_base_dir=None):
    """
    Ensure Stable Diffusion 2.1 base config files exist under diffusers/stable-diffusion-2-1-base.
    Downloads missing config files from Manojb/stable-diffusion-2-1-base on Hugging Face.
    """
    diffusers_paths = folder_paths.get_folder_paths("diffusers") or []
    if sd21_base_dir is None:
        if not diffusers_paths:
            # fallback to ComfyUI models_dir/diffusers
            sd21_base_dir = os.path.join(folder_paths.models_dir, "diffusers", "stable-diffusion-2-1-base")
        else:
            sd21_base_dir = os.path.join(diffusers_paths[0], "stable-diffusion-2-1-base")

    os.makedirs(sd21_base_dir, exist_ok=True)

    # check which files are missing
    missing = []
    for rel_path in SD21_MANOJB_FILES.keys():
        target = os.path.join(sd21_base_dir, rel_path)
        if not os.path.isfile(target):
            missing.append(rel_path)

    if not missing:
        print(f"[SDMatte] SD 2.1 configs already present at: {sd21_base_dir}")
        return sd21_base_dir

    base_url = "https://huggingface.co/Manojb/stable-diffusion-2-1-base/resolve/main"
    try:
        import requests
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None
    except Exception:
        requests = None
        tqdm = None

    for rel_path in missing:
        url = f"{base_url}/{rel_path}"
        target = os.path.join(sd21_base_dir, rel_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"[SDMatte] Downloading {rel_path} from Manojb -> {target}")
        try:
            if requests:
                tmp = target + ".tmp"
                resp = requests.get(url, stream=True, timeout=60)
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0) or 0)
                with open(tmp, "wb") as f:
                    bar = None
                    if tqdm and total > 0:
                        bar = tqdm(desc=rel_path, total=total, unit="iB", unit_scale=True, unit_divisor=1024)
                    for chunk in resp.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            if bar:
                                bar.update(len(chunk))
                    if bar:
                        bar.close()
                os.replace(tmp, target)
            else:
                import urllib.request
                urllib.request.urlretrieve(url, target)
            print(f"[SDMatte] Downloaded {rel_path}")
        except Exception as e:
            print(f"[SDMatte] Warning: failed to download {rel_path}: {e}")

    return sd21_base_dir

def download_model(model_name, models_dir=MODEL_DIR, model_urls=MODEL_URLS):
    # 1) Search in all registered SDMatte paths first
    all_search_paths = folder_paths.get_folder_paths("SDMatte") or []
    for search_path in all_search_paths:
        check_path = os.path.join(search_path, model_name)
        if os.path.isfile(check_path):
            try:
                if os.path.getsize(check_path) > 0:
                    print(f"[SDMatte] Found model at: {check_path}")
                    return check_path
            except OSError:
                pass  # couldn't stat; continue

    # 2) Not found -> prepare to download to models_dir
    url = model_urls.get(model_name)
    if not url:
        raise ValueError(f"[SDMatte] Unknown model name: {model_name}")

    target_path = os.path.join(models_dir, model_name)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # if target exists and non-empty, use it
    if os.path.isfile(target_path):
        try:
            if os.path.getsize(target_path) > 0:
                return target_path
        except OSError:
            pass

    print(f"[SDMatte] Model '{model_name}' not found. Downloading to {target_path}...")

    tmp_path = target_path + ".tmp"

    try:
        try:
            import requests
            try:
                from tqdm import tqdm  # optional
            except Exception:
                tqdm = None

            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0) or 0)

                with open(tmp_path, 'wb') as f:
                    bar = None
                    if tqdm and total_size > 0:
                        bar = tqdm(desc=model_name, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024)

                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            if bar:
                                bar.update(len(chunk))

                    if bar:
                        bar.close()

            # optional size check
            if total_size > 0:
                try:
                    if os.path.getsize(tmp_path) != total_size:
                        raise IOError(f"[SDMatte] Incomplete download: {os.path.getsize(tmp_path)} != {total_size}")
                except OSError:
                    raise

        except (ImportError, ModuleNotFoundError):
            import urllib.request
            urllib.request.urlretrieve(url, tmp_path)

        # concurrent safety: if another process already finished
        if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return target_path

        os.replace(tmp_path, target_path)  # atomic
        print(f"[SDMatte] Download complete: {target_path}")
        return target_path

    except KeyboardInterrupt:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

SDMatteCore = None


def _resize_norm_image_bchw(image_bchw: torch.Tensor, size_hw=(1024, 1024)) -> torch.Tensor:
    resize = transforms.Resize(size_hw, antialias=True)
    norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    x = resize(image_bchw)
    x = norm(x)
    return x


def _resize_mask_b1hw(mask_b1hw: torch.Tensor, size_hw=(1024, 1024)) -> torch.Tensor:
    resize = transforms.Resize(size_hw)
    return resize(mask_b1hw)


class SDMatteApply:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(MODEL_URLS.keys()), ),
                "image": ("IMAGE", {"tooltip": "需要进行抠图的输入图像"}),
                "trimap": ("MASK", {"tooltip": "三值图掩码：白色=前景，黑色=背景，灰色=未知区域"}),
                "inference_size": ([512, 640, 768, 896, 1024], {
                    "default": 1024, 
                    "tooltip": "推理分辨率，越高质量越好但速度越慢。推荐1024(最高质量)或768(平衡性能)"
                }),
                "is_transparent": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "输入图像是否包含透明通道。如果原图有透明背景请启用"
                }),
                "output_mode": (["alpha_only", "matted_rgba", "matted_rgb"], {
                    "default": "alpha_only",
                    "tooltip": "输出模式：alpha_only=只输出遮罩；matted_rgba=透明背景抠图；matted_rgb=黑色背景抠图(推荐，避免干扰)"
                }),
                "mask_refine": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "启用遮罩优化，使用trimap约束过滤不需要的区域，减少背景干扰"
                }),
                "trimap_constraint": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1,
                    "tooltip": "trimap约束强度(0.1-1.0)。越高约束越严格，0.8=平衡，0.9=严格过滤，0.6=宽松保留"
                }),
            },
            "optional": {
                "force_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("alpha_mask", "matted_image")
    FUNCTION = "apply_matte"
    CATEGORY = "Matting/SDMatte"

    def apply_matte(self, ckpt_name, image, trimap, inference_size, is_transparent, output_mode, mask_refine, trimap_constraint, force_cpu=False):
        device = comfy.model_management.get_torch_device()
        if force_cpu:
            device = torch.device('cpu')

        global SDMatteCore
        if SDMatteCore is None:
            from .src.modeling.SDMatte.meta_arch import SDMatte as SDMatteCore

        diffusers_paths = folder_paths.get_folder_paths("diffusers") or []
        pretrained_repo = None
        for path in diffusers_paths:
            candidate_path = os.path.join(path, "stable-diffusion-2-1-base")
            if os.path.isdir(candidate_path):
                pretrained_repo = candidate_path
                break

        if pretrained_repo is None:
            # not found locally — try downloading the config-only files from Manojb
            try:
                sd21_dir = ensure_sd21_from_manojb()
                if sd21_dir and os.path.isdir(sd21_dir):
                    pretrained_repo = sd21_dir
            except Exception as e:
                print(f"[SDMatte] Warning: failed to auto-download SD 2.1 configs: {e}")

        if pretrained_repo is None:
            raise FileNotFoundError("Stable Diffusion 2.1 base model not found in diffusers directory and auto-download failed. Please provide configs at diffusers/stable-diffusion-2-1-base or ensure network access.")
        
        sdmatte_model = SDMatteCore(
            pretrained_model_name_or_path=pretrained_repo,
            load_weight=False,
            use_aux_input=True,
            aux_input="trimap",
            aux_input_list=["point_mask", "bbox_mask", "mask", "trimap"],
            attn_mask_aux_input=["point_mask", "bbox_mask", "mask", "trimap"],
            use_encoder_hidden_states=True,
            use_attention_mask=True,
            add_noise=False,
        )
        
        ckpt_path = download_model(ckpt_name)
        
        from safetensors import safe_open
        state_dict = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        state_root = state_dict

        candidate_keys = [
            'state_dict','model_state_dict','params','weights',
            'ema','model_ema','ema_state_dict','net','module','model','unet'
        ]
        state_dict = None
        if isinstance(state_root, dict):
            for k in candidate_keys:
                inner = state_root.get(k)
                if isinstance(inner, dict):
                    state_dict = inner
                    break
        if state_dict is None:
            state_dict = state_root

        sdmatte_model.load_state_dict(state_dict, strict=False)
        sdmatte_model.eval()
        sdmatte_model.to(device)

        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            
            try:
                unet = getattr(sdmatte_model, 'unet', None)
                if unet is not None and hasattr(unet, 'set_attn_processor'):
                    from diffusers.models.attention_processor import SlicedAttnProcessor
                    unet.set_attn_processor(SlicedAttnProcessor(slice_size=1))
            except Exception:
                pass

        B, H, W, C = image.shape
        orig_h, orig_w = H, W

        img_bchw = image.permute(0, 3, 1, 2).contiguous().to(device)
        img_in = _resize_norm_image_bchw(img_bchw, (int(inference_size), int(inference_size)))

        is_trans = torch.tensor([1 if is_transparent else 0] * B, device=device)
        data = {"image": img_in, "is_trans": is_trans, "caption": [""] * B}

        def to_b1hw(x):
            return _resize_mask_b1hw(x.unsqueeze(1).contiguous().to(device), (int(inference_size), int(inference_size)))

        tri = to_b1hw(trimap) * 2 - 1
        data["trimap"] = tri
        data["trimap_coords"] = torch.tensor([[0,0,1,1]]*B, dtype=tri.dtype, device=device)

        with torch.no_grad():
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_alpha = sdmatte_model(data)
            else:
                pred_alpha = sdmatte_model(data)

        out = transforms.Resize((orig_h, orig_w))(pred_alpha)
        out = out.squeeze(1).clamp(0, 1).detach().cpu()
        
        if mask_refine:
            trimap_cpu = trimap.cpu()
            
            foreground_regions = trimap_cpu > trimap_constraint
            background_regions = trimap_cpu < (1.0 - trimap_constraint)
            unknown_regions = ~(foreground_regions | background_regions)
            
            refined_alpha = out.clone()
            refined_alpha[background_regions] = 0.0
            refined_alpha[foreground_regions] = torch.clamp(refined_alpha[foreground_regions] * 1.2, 0, 1)
            
            alpha_threshold = 0.3
            low_confidence = (refined_alpha < alpha_threshold) & unknown_regions
            refined_alpha[low_confidence] = 0.0
            
            out = refined_alpha
        
        alpha_expanded = out.unsqueeze(-1)
        
        if output_mode == "alpha_only":
            matted_image = torch.zeros_like(image.cpu())
        elif output_mode == "matted_rgba":
            matted_image = torch.cat([
                image.cpu(),
                alpha_expanded.expand(-1, -1, -1, 1)
            ], dim=-1)
        elif output_mode == "matted_rgb":
            trimap_cpu = trimap.cpu()
            trimap_expanded = trimap_cpu.unsqueeze(-1)
            foreground_mask = (trimap_expanded > 0.2) & (alpha_expanded > 0.1)
            matted_image = image.cpu() * foreground_mask.float()
        else:
            matted_image = image.cpu() * alpha_expanded
        
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return (out, matted_image)


NODE_CLASS_MAPPINGS = {
    "SDMatteApply": SDMatteApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDMatteApply": "Apply SDMatte",
}


