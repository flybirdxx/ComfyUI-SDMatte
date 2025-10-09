"""
Inference for Composition-1k Dataset.

Run:
python inference.py \
    --config-dir path/to/config
    --checkpoint-dir path/to/checkpoint
    --inference-dir path/to/inference
    --data-dir path/to/data
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from os.path import join
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser
import warnings
from data.dataset import DataGenerator
import time

warnings.filterwarnings("ignore")


# model and output
def matting_inference(
    config_dir="",
    checkpoint_dir="",
    inference_dir="",
    setname="",
):
    # initializing model
    torch.set_grad_enabled(False)
    cfg = LazyConfig.load(config_dir)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    DetectionCheckpointer(model).load(checkpoint_dir)
    model.eval()

    # initializing dataset
    test_dataset = DataGenerator(set_list=setname, phase="test", psm=cfg.hy_dict.psm, radius=cfg.hy_dict.radius)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)

    # inferencing
    os.makedirs(inference_dir, exist_ok=True)
    # image_dir = inference_dir.replace(args.setname, args.setname + "_Image")
    # os.makedirs(image_dir, exist_ok=True)

    start_time = time.time()
    for data in tqdm(test_loader):
        image_name = data["image_name"][0]
        H, W = data["hw"][0].item(), data["hw"][1].item()

        # # image
        # image = data["image"]
        # image = (image + 1) / 2
        # image = image.squeeze(0).permute(1, 2, 0) * 255
        # image = cv2.resize(image.detach().cpu().numpy(), (W, H)).astype(np.uint8)
        # red = np.array([[255.0, 0.0, 0.0]], dtype=image.dtype)
        # blue = np.array([[0, 0.0, 255.0]], dtype=image.dtype)
        # if cfg.hy_dict.model_kwargs.aux_input == "bbox_mask":
        #     coords = data["bbox_coords"][0]
        #     x_min, y_min, x_max, y_max = (
        #         int(coords[0].item() * W),
        #         int(coords[1].item() * H),
        #         int(coords[2].item() * W),
        #         int(coords[3].item() * H),
        #     )
        #     image[y_min:y_max, x_min, :] = red
        #     image[y_min:y_max, x_max, :] = red
        #     image[y_min, x_min:x_max, :] = red
        #     image[y_max, x_min:x_max, :] = red
        # elif cfg.hy_dict.model_kwargs.aux_input == "point_mask":
        #     point_mask = data["point_mask"]
        #     point_mask = (point_mask + 1) / 2
        #     # point_mask = point_mask * 255.0
        #     point_mask = cv2.resize(point_mask.flatten(0, 2).detach().cpu().numpy(), (W, H), interpolation=cv2.INTER_NEAREST)
        #     if len(np.unique(point_mask)) == 3:
        #         image[point_mask == 0] = red
        #         image[point_mask == 1] = blue
        #     elif len(np.unique(point_mask)) == 2:
        #         image[point_mask == 1] = red
        # image = F.to_pil_image(image).convert("RGB")
        # image.save(join(image_dir, image_name[:-4] + ".jpg"))

        with torch.no_grad():
            pred = model(data)
            output = pred.flatten(0, 2) * 255
            output = cv2.resize(output.detach().cpu().numpy(), (W, H)).astype(np.uint8)
            output = F.to_pil_image(output).convert("RGB")
            output.save(join(inference_dir, image_name))
            torch.cuda.empty_cache()
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(test_loader)

    print(f"Start time: {start_time:.4f} seconds")
    print(f"End time: {end_time:.4f} seconds")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per iteration: {avg_time:.4f} seconds")


if __name__ == "__main__":
    # add argument we need:
    parser = default_argument_parser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--inference-dir", type=str, required=True)
    parser.add_argument("--setname", type=str, required=True)

    args = parser.parse_args()
    matting_inference(
        config_dir=args.config_dir,
        checkpoint_dir=args.checkpoint_dir,
        inference_dir=args.inference_dir,
        setname=args.setname,
    )
