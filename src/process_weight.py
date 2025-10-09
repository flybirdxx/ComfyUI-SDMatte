from safetensors.torch import load_file, save_file
import torch

# 加载 .safetensors 权重
weights_path = (
    "stabilityai-stable-diffusion-2/unet/diffusion_pytorch_model.safetensors"
)
state_dict = load_file(weights_path)

# 定义缺失键和对应维度
missing_keys = [
    "bbox_embedding.linear_1.weight",
    "bbox_embedding.linear_1.bias",
    "bbox_embedding.linear_2.weight",
    "bbox_embedding.linear_2.bias",
    "point_embedding.linear_2.weight",
    "point_embedding.linear_1.weight",
    "point_embedding.linear_1.bias",
    "point_embedding.linear_2.bias",
]
point_dim = 1680
bbox_dim = 1280  # 示例，实际需要根据模型架构确定
output_dim = 1280  # 示例，实际需要根据模型架构确定

# 将新增权重初始化为零
for key in missing_keys:
    if "_1.weight" in key and "bbox" in key:
        state_dict[key] = torch.zeros((output_dim, bbox_dim))  # 初始化权重为零
    elif "_2.weight" in key and "bbox" in key:
        state_dict[key] = torch.zeros((output_dim, output_dim))  # 初始化权重为零
    elif "_1.weight" in key and "point" in key:
        state_dict[key] = torch.zeros((output_dim, point_dim))
    elif "_2.weight" in key and "point" in key:
        state_dict[key] = torch.zeros((output_dim, output_dim))  # 初始化权重为零
    elif "bias" in key:
        state_dict[key] = torch.zeros(output_dim)  # 初始化偏置为零

# 保存更新后的权重文件
new_weights_path = (
    "stabilityai-stable-diffusion-2/unet/diffusion_pytorch_model.safetensors"
)
save_file(state_dict, new_weights_path)

print(f"Updated weights saved to {new_weights_path}")
