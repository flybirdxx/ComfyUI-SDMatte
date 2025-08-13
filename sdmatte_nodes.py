import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

import folder_paths
import comfy
from comfy.utils import load_torch_file

# 延迟导入核心模型，避免在依赖未安装时阻断节点注册
SDMatteCore = None


class SDMatteModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }

    RETURN_TYPES = ("SDMATTE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Matting/SDMatte"

    def load_model(self, ckpt_name):
        device = comfy.model_management.get_torch_device()

        global SDMatteCore
        if SDMatteCore is None:
            from .src.modeling.SDMatte.meta_arch import SDMatte as SDMatteCore  # type: ignore

        # 使用本地固定路径作为 diffusers 基底
        base_dir = os.path.dirname(__file__)
        pretrained_repo = os.path.join(base_dir, "src", "SDMatte")
        required_subdirs = ["text_encoder", "vae", "unet", "scheduler", "tokenizer"]
        missing = [d for d in required_subdirs if not os.path.isdir(os.path.join(pretrained_repo, d))]
        if missing:
            raise FileNotFoundError(
                f"本地基底缺少目录: {missing}. 期望路径: {pretrained_repo}，需包含 {required_subdirs} 子目录。"
            )

        print(f"[SDMatte] Using base repo: {pretrained_repo}")
        # 实例化模型（使用默认优化配置）
        model = SDMatteCore(
            pretrained_model_name_or_path=pretrained_repo,
            load_weight=False,
            use_aux_input=True,
            aux_input="trimap",
            use_encoder_hidden_states=True,
            use_attention_mask=True,
            add_noise=False,
        )

        # 加载合并后的权重
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        print(f"[SDMatte] Loading merged weights: {ckpt_path}")
        # 更健壮的加载与 state_dict 提取
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

        # 尝试从常见键中提取真正权重
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
            # 如果顶层就是大量权重键
            if len(state_root) > 50 and any('.weight' in x or '.bias' in x for x in state_root.keys()):
                state_dict = state_root
        if state_dict is None:
            top_keys = list(state_root.keys()) if isinstance(state_root, dict) else []
            raise RuntimeError(
                f"无法从权重中提取 state_dict。顶层键: {top_keys[:10]} (共{len(top_keys)})。请使用官方脚本导出仅模型权重的 sdmatte.pth。"
            )

        print(f"[SDMatte] Weights loaded, keys: {len(list(state_dict.keys()))}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[SDMatte] load_state_dict done. missing={len(missing)}, unexpected={len(unexpected)}")

        model.eval()
        model.to(device)

        return (model,)


def _resize_norm_image_bchw(image_bchw: torch.Tensor, size_hw=(1024, 1024)) -> torch.Tensor:
    # image_bchw in [0,1], float32
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
                "sdmatte_model": ("SDMATTE_MODEL",),
                "image": ("IMAGE",),
                "trimap": ("MASK",),
                "inference_size": ([512, 640, 768, 896, 1024], {"default": 1024}),
                "is_transparent": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "force_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_matte"
    CATEGORY = "Matting/SDMatte"

    def apply_matte(self, sdmatte_model, image, trimap, inference_size, is_transparent, force_cpu=False):
        device = comfy.model_management.get_torch_device()
        if force_cpu:
            device = torch.device('cpu')
        sdmatte_model.to(device)

        # 自动显存优化（内置默认设置）
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()  # 预清理显存
            except Exception:
                pass
            
            # 启用注意力切片以节省显存
            try:
                unet = getattr(sdmatte_model, 'unet', None)
                if unet is not None and hasattr(unet, 'set_attn_processor'):
                    from diffusers.models.attention_processor import SlicedAttnProcessor
                    # 使用适中的切片大小平衡性能和显存
                    unet.set_attn_processor(SlicedAttnProcessor(slice_size=1))
                    print('[SDMatte] 已启用注意力切片优化')
            except Exception as e:
                print(f'[SDMatte] 注意力优化跳过: {e}')

        B, H, W, C = image.shape
        orig_h, orig_w = H, W

        # IMAGE: (B,H,W,C)->(B,C,H,W) in [0,1]
        img_bchw = image.permute(0, 3, 1, 2).contiguous().to(device)
        img_in = _resize_norm_image_bchw(img_bchw, (int(inference_size), int(inference_size)))

        # 构造必要的数据字典字段
        is_trans = torch.tensor([1 if is_transparent else 0] * B, device=device)

        data = {"image": img_in, "is_trans": is_trans, "caption": [""] * B}

        def to_b1hw(x):
            return _resize_mask_b1hw(x.unsqueeze(1).contiguous().to(device), (int(inference_size), int(inference_size)))

        # 处理 trimap 输入
        tri = to_b1hw(trimap) * 2 - 1
        data["trimap"] = tri
        data["trimap_coords"] = torch.tensor([[0,0,1,1]]*B, dtype=tri.dtype, device=device)

        with torch.no_grad():
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred_alpha = sdmatte_model(data)
            else:
                pred_alpha = sdmatte_model(data)

        # 还原到原尺寸
        out = transforms.Resize((orig_h, orig_w))(pred_alpha)
        out = out.squeeze(1).clamp(0, 1).detach().cpu()

        # 推理后清理显存
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return (out,)


NODE_CLASS_MAPPINGS = {
    "SDMatteModelLoader": SDMatteModelLoader,
    "SDMatteApply": SDMatteApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDMatteModelLoader": "Load SDMatte Model",
    "SDMatteApply": "Apply SDMatte",
}


