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
                "sdmatte_model": ("SDMATTE_MODEL", {"tooltip": "已加载的SDMatte模型，来自SDMatte Model Loader节点"}),
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

    def apply_matte(self, sdmatte_model, image, trimap, inference_size, is_transparent, output_mode, mask_refine, trimap_constraint, force_cpu=False):
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
        
        # 遮罩优化：使用trimap约束来过滤不需要的区域
        if mask_refine:
            trimap_cpu = trimap.cpu()  # (B, H, W)
            
            # 创建严格的前景约束
            # 只在trimap明确标记为前景(白色区域)的地方保留高alpha值
            # 在未知区域(灰色)允许模型输出，但在背景区域(黑色)强制为0
            foreground_regions = trimap_cpu > trimap_constraint  # 前景区域
            background_regions = trimap_cpu < (1.0 - trimap_constraint)  # 背景区域
            unknown_regions = ~(foreground_regions | background_regions)  # 未知区域
            
            # 优化alpha遮罩
            refined_alpha = out.clone()
            
            # 背景区域强制为0
            refined_alpha[background_regions] = 0.0
            
            # 前景区域保持原值或增强
            refined_alpha[foreground_regions] = torch.clamp(refined_alpha[foreground_regions] * 1.2, 0, 1)
            
            # 未知区域保持模型输出，但应用阈值过滤
            alpha_threshold = 0.3  # 过滤掉低置信度的区域
            low_confidence = (refined_alpha < alpha_threshold) & unknown_regions
            refined_alpha[low_confidence] = 0.0
            
            out = refined_alpha
        
        # 根据输出模式创建不同的抠图结果
        # out: (B, H, W), image: (B, H, W, C)
        alpha_expanded = out.unsqueeze(-1)  # (B, H, W, 1)
        
        if output_mode == "alpha_only":
            # 只输出alpha遮罩，图像输出为黑色
            matted_image = torch.zeros_like(image.cpu())
        elif output_mode == "matted_rgba":
            # 创建RGBA图像：RGB通道保持原图，A通道为alpha遮罩
            # 背景变为透明，只保留前景对象
            matted_image = torch.cat([
                image.cpu() * alpha_expanded,  # RGB通道应用遮罩
                alpha_expanded.expand(-1, -1, -1, 1)  # A通道为遮罩本身
            ], dim=-1)
            # 确保输出仍为3通道（ComfyUI IMAGE格式）
            matted_image = matted_image[..., :3]
        elif output_mode == "matted_rgb":
            # 只保留前景对象，背景变为黑色
            # 使用trimap的前景信息来更精确地提取对象
            trimap_cpu = trimap.cpu()  # (B, H, W)
            trimap_expanded = trimap_cpu.unsqueeze(-1)  # (B, H, W, 1)
            
            # 只在trimap标记为前景(>0.8)或未知区域(0.2-0.8)的地方保留原图
            # 结合alpha遮罩进一步细化
            foreground_mask = (trimap_expanded > 0.2) & (alpha_expanded > 0.1)
            matted_image = image.cpu() * foreground_mask.float()
        else:
            # 默认：直接应用alpha遮罩
            matted_image = image.cpu() * alpha_expanded
        
        # 推理后清理显存
        if device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return (out, matted_image)


NODE_CLASS_MAPPINGS = {
    "SDMatteModelLoader": SDMatteModelLoader,
    "SDMatteApply": SDMatteApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDMatteModelLoader": "Load SDMatte Model",
    "SDMatteApply": "Apply SDMatte",
}


