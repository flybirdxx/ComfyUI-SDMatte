import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import folder_paths
import comfy
from comfy.utils import load_torch_file
def download_with_progress(url, filepath, description="File"):
    try:
        import requests
        from tqdm import tqdm
        
        print(f"[SDMatte] Starting download: {description}")
        print(f"[SDMatte] URL: {url}")
        print(f"[SDMatte] Save path: {filepath}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {description}"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
        
        print(f"[SDMatte] Download completed: {filepath}")
        return True
        
    except Exception as e:
        print(f"[SDMatte] Download failed: {e}")
        return False
def verify_file(filepath):
    if not os.path.exists(filepath):
        return False
    
    actual_size = os.path.getsize(filepath)
    if actual_size > 0:
        print(f"[SDMatte] File verification successful, size: {actual_size} bytes")
        return True
    
    return False
SDMatteCore = None
SDMATTE_MODEL_LIST = [
    "SDMatte.pth",
    "SDMatte_plus.pth"
]
SDMATTE_MODEL_MAPPING = {
    "SDMatte.pth": {
        "url": "https://huggingface.co/LongfeiHuang/SDMatte/resolve/main/SDMatte.pth",
        "filename": "SDMatte.pth",
        "description": "SDMatte Standard Version (12.2 GB)"
    },
    "SDMatte_plus.pth": {
        "url": "https://huggingface.co/LongfeiHuang/SDMatte/resolve/main/SDMatte_plus.pth",
        "filename": "SDMatte_plus.pth", 
        "description": "SDMatte Plus Version (12.1 GB)"
    }
}
_sdmatte_model_cache = {
    'model': None,
    'ckpt_name': None,
}
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
def get_model_path(ckpt_name):
    sdmatte_dir = os.path.join(folder_paths.models_dir, "SDMatte")
    os.makedirs(sdmatte_dir, exist_ok=True)
    
    local_model_path = os.path.join(sdmatte_dir, ckpt_name)
    
    if os.path.exists(local_model_path):
        print(f"[SDMatte] Using local model from models/SDMatte/: {local_model_path}")
        return local_model_path
    
    if ckpt_name not in SDMATTE_MODEL_MAPPING:
        print(f"[SDMatte] User custom model path: {local_model_path}")
        return local_model_path
    
    model_info = SDMATTE_MODEL_MAPPING[ckpt_name]
    print(f"[SDMatte] Local model {ckpt_name} not found, preparing to download from HuggingFace...")
    print(f"[SDMatte] Model description: {model_info['description']}")
    
    try:
        success = download_with_progress(
            url=model_info["url"],
            filepath=local_model_path,
            description=model_info["description"]
        )
        
        if success and verify_file(local_model_path):
            print(f"[SDMatte] Model download and verification successful: {local_model_path}")
            return local_model_path
        else:
            raise FileNotFoundError(f"Model download or verification failed: {local_model_path}")
            
    except Exception as e:
        print(f"[SDMatte] Auto download failed: {e}")
        if os.path.exists(local_model_path):
            try:
                os.remove(local_model_path)
                print(f"[SDMatte] Cleaned up incomplete download file: {local_model_path}")
            except:
                pass
        raise FileNotFoundError(f"Unable to get model {ckpt_name}: {e}")
def load_sdmatte_model(ckpt_name, force_cpu=False):
    global _sdmatte_model_cache, SDMatteCore
    
    if (_sdmatte_model_cache['model'] is not None and 
        _sdmatte_model_cache['ckpt_name'] == ckpt_name):
        print(f"[SDMatte] Reusing cached model: {ckpt_name}")
        return _sdmatte_model_cache['model']
    
    if _sdmatte_model_cache['model'] is not None:
        print("[SDMatte] Cleaning up old model cache")
        del _sdmatte_model_cache['model']
        clear_memory()
    
    device = comfy.model_management.get_torch_device()
    if force_cpu:
        device = torch.device('cpu')
    if SDMatteCore is None:
        from .src.modeling.SDMatte.meta_arch import SDMatte as SDMatteCore  # type: ignore
    base_dir = os.path.dirname(__file__)
    pretrained_repo = os.path.join(base_dir, "src", "SDMatte")
    required_subdirs = ["text_encoder", "vae", "unet", "scheduler", "tokenizer"]
    missing = [d for d in required_subdirs if not os.path.isdir(os.path.join(pretrained_repo, d))]
    if missing:
        raise FileNotFoundError(
            f"Local base missing directories: {missing}. Expected path: {pretrained_repo}, must contain {required_subdirs} subdirectories."
        )
    print(f"[SDMatte] Using base repo: {pretrained_repo}")
    model = SDMatteCore(
        pretrained_model_name_or_path=pretrained_repo,
        load_weight=False,
        use_aux_input=True,
        aux_input="trimap",
        use_encoder_hidden_states=True,
        use_attention_mask=True,
        add_noise=False,
    )
    ckpt_path = get_model_path(ckpt_name)
    print(f"[SDMatte] Loading merged weights: {ckpt_path}")
    state_root = None
    try:
        from torch.serialization import add_safe_globals
        try:
            from omegaconf.listconfig import ListConfig
            add_safe_globals([ListConfig])
        except Exception:
            pass
        try:
            from omegaconf.base import ContainerMetadata
            add_safe_globals([ContainerMetadata])
        except Exception:
            pass
        try:
            state_root = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            print("[SDMatte] torch.load(weights_only=True) ok")
        except Exception as e:
            print(f"[SDMatte] weights_only=True failed: {e}; retry weights_only=False")
            try:
                state_root = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                print("[SDMatte] torch.load(weights_only=False) ok")
            except TypeError:
                state_root = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"[SDMatte] torch.load failed: {e}; fallback comfy.load_torch_file")
        state_root = load_torch_file(ckpt_path)
    candidate_keys = [
        'state_dict','model_state_dict','params','weights',
        'ema','model_ema','ema_state_dict','net','module','model','unet'
    ]
    state_dict = None
    if isinstance(state_root, dict):
        for k in candidate_keys:
            inner = state_root.get(k)
            if isinstance(inner, dict) and len(inner) > 50:
                state_dict = inner
                print(f"[SDMatte] extracted inner dict via key: {k}")
                break
    if state_dict is None and isinstance(state_root, dict):
        if len(state_root) > 50 and any('.weight' in x or '.bias' in x for x in state_root.keys()):
            state_dict = state_root
    if state_dict is None:
        top_keys = list(state_root.keys()) if isinstance(state_root, dict) else []
        raise RuntimeError(
            f"Unable to extract state_dict from weights. Top-level keys: {top_keys[:10]} (total {len(top_keys)}). Please use official script to export model-only weights to sdmatte.pth."
        )
    print(f"[SDMatte] Weights loaded, keys: {len(list(state_dict.keys()))}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[SDMatte] load_state_dict done. missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    model.to(device)
    
    _sdmatte_model_cache['model'] = model
    _sdmatte_model_cache['ckpt_name'] = ckpt_name
    
    return model
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
                "ckpt_name": (SDMATTE_MODEL_LIST, {
                    "default": "SDMatte.pth",
                    "tooltip": "选择SDMatte模型：SDMatte.pth(标准版) 或 SDMatte_plus.pth(增强版)。模型将存储在models/SDMatte/目录，如果本地不存在将自动下载"
                }),
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
                    "default": "matted_rgba",
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
                "unload_model_after": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "推理完成后卸载模型释放显存。适用于显存紧张的情况，但会影响后续推理速度"
                }),
            },
            "optional": {
                "force_cpu": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "强制使用CPU推理。仅在GPU显存不足时启用，会显著降低速度"
                }),
            },
        }
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("alpha_mask", "matted_image")
    FUNCTION = "apply_matte"
    CATEGORY = "Matting/SDMatte"
    def apply_matte(self, ckpt_name, image, trimap, inference_size, is_transparent, output_mode, mask_refine, trimap_constraint, unload_model_after=False, force_cpu=False):
        sdmatte_model = load_sdmatte_model(ckpt_name, force_cpu)
        
        device = comfy.model_management.get_torch_device()
        if force_cpu:
            device = torch.device('cpu')
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
            except Exception as e:
                print(f'[SDMatte] Attention optimization skipped: {e}')
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
                image.cpu() * alpha_expanded,
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
        
        if unload_model_after:
            global _sdmatte_model_cache
            print("[SDMatte] Unloading model to free GPU memory")
            if _sdmatte_model_cache['model'] is not None:
                del _sdmatte_model_cache['model']
                _sdmatte_model_cache['model'] = None
                _sdmatte_model_cache['ckpt_name'] = None
            clear_memory()
        return (out, matted_image)
NODE_CLASS_MAPPINGS = {
    "SDMatteApply": SDMatteApply,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDMatteApply": "Apply SDMatte",
}
