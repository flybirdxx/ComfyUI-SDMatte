import os
import numpy as np
import torch
import cv2
import json
import random
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms
from scipy.ndimage import label
import scipy.ndimage

cv2.setNumThreads(1)

DATA_TEST_PATH = {
    "AIM-500": "AIM-500/",
    "AM-2K": "AM-2K/validation/",
    "P3M-500-NP": "P3M-10k/validation/P3M-500-NP/",
    "UHRSD": "UHRSD/UHRSD_TE_2K/",
    "Trans-460": "Transparent-460/Test/",
    "RefMatte_RW_100": "RefMatte_RW_100/",
}

DATA_TEST_ARGS = {
    "AIM-500": ["original", "mask", "trimap"],
    "AM-2K": ["original", "mask", "trimap"],
    "P3M-500-NP": ["original_image", "mask", "trimap"],
    "UHRSD": ["image", "mask"],
    "Trans-460": ["fg", "alpha", "trimap"],
    "RefMatte_RW_100": ["image", "mask", "refmatte_rw100_label.json"],
}

DATA_TRAIN_PATH = {
    "Composition-1K": "Combined_Dataset/Training_set/",
    "DIS-646": "Distinctions-646/Train",
    "P3M-10K": "P3M-10k/train/",
    "AM-2K": "AM-2K/train/",
    "UHRSD": "UHRSD/UHRSD_TR_2K/",
    "RWP-636": "RealWorldPortrait-636/",
    "Trans-460": "Transparent-460/Train/",
    "COCO_Matting": "COCO_Matting/COCO_Matte.json",
    "RefMatte": "RefMatte/RefMatte/train/",
}

DATA_TRAIN_ARGS = {
    "Composition-1K": ["combined_fg", "combined_alpha", 3],
    "DIS-646": ["FG", "GT", 3],
    "P3M-10K": ["blurred_image", "mask"],
    "AM-2K": ["original", "mask", "fg", 4],
    "UHRSD": ["image", "mask"],
    "RWP-636": ["image", "alpha"],
    "Trans-460": ["fg", "alpha", 5],
    "RefMatte": ["img", "mask", "refmatte_train_label.json"],
}

BG_PATH = "BG-20K/"
BG_LIST = os.listdir(BG_PATH)


def generate_samples_from_refmatte(image_folder_path, alpha_folder_path, json_path):
    alpha_list = os.listdir(alpha_folder_path)
    paths_list = []
    with open(json_path, "r") as file:
        data_json = json.load(file)

    for alpha_name in alpha_list:
        if not alpha_name.endswith(".png"):
            continue
        path = {}
        alpha_path = os.path.join(alpha_folder_path, alpha_name)
        alpha_name, _ = os.path.splitext(alpha_name)
        image_name = data_json[alpha_name]["image_name"] + ".jpg"
        image_path = os.path.join(image_folder_path, image_name)
        caption = data_json[alpha_name]["expressions"][-1]
        path["image"] = image_path
        path["alpha"] = alpha_path
        path["caption"] = caption
        path["is_trans"] = 0
        paths_list.append(path)
    return paths_list


def generate_samples_from_refmatte_10k(image_path, alpha_path):
    with open(image_path, "r") as file:
        image_list = [line.strip() for line in file]
    with open(alpha_path, "r") as file:
        alpha_list = [line.strip() for line in file]
    paths_list = []

    for i in range(len(alpha_list)):
        if not alpha_list[i].endswith(".png"):
            continue
        path = {}
        path["image"] = image_list[i]
        path["alpha"] = alpha_list[i]
        path["caption"] = ""
        path["is_trans"] = 0
        paths_list.append(path)
    return paths_list


def generate_samples_from_coco_matte(json_path):
    paths_list = []
    with open(json_path, "r") as file:
        data = json.load(file)

    for item in data:
        path = {}
        alpha_path = item["alpha_path"]
        image_path = item["image_path"]
        trimap_path = alpha_path.replace("alpha", "trimap")
        path["image"] = image_path
        path["alpha"] = alpha_path
        path["trimap"] = trimap_path
        path["is_trans"] = 0
        paths_list.append(path)
    return paths_list


class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]

    def __call__(self, sample):
        alpha = sample["alpha"]
        trimap = sample["trimap"]

        if trimap is not None:
            size = alpha.shape[::-1]
            trimap = cv2.resize(trimap, size, interpolation=cv2.INTER_NEAREST)
        else:
            ### generate trimap from alpha
            fg_width = np.random.randint(15, 30)
            bg_width = np.random.randint(15, 30)
            fg_mask = alpha + 1e-5
            bg_mask = 1 - alpha + 1e-5
            fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
            bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

            trimap = np.ones_like(alpha) * 0.5
            trimap[fg_mask > 0.95] = 1.0
            trimap[bg_mask > 0.95] = 0.0

        sample["trimap"] = trimap.astype(np.float32)

        return sample


class GenMask(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]
        self.max_kernel_size = 30
        self.min_kernel_size = 15

    def __call__(self, sample):
        alpha = sample["alpha"]
        mask = sample["mask"]
        h, w = alpha.shape
        if mask is not None:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            ### generate mask
            low = 0.01
            high = 1.0
            thres = random.random() * (high - low) + low
            seg_mask = (alpha >= thres).astype(np.int_).astype(np.uint8)
            random_num = random.randint(0, 3)
            if random_num == 0:
                seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])
            elif random_num == 1:
                seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])
            elif random_num == 2:
                seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])
                seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])
            elif random_num == 3:
                seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])
                seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(self.min_kernel_size, self.max_kernel_size)])

            mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        coords = np.nonzero(mask)
        if coords[0].size == 0 or coords[1].size == 0:
            mask_coords = np.array([0, 0, 1, 1])
        else:
            y_min, x_min = np.argwhere(mask).min(axis=0)
            y_max, x_max = np.argwhere(mask).max(axis=0)
            y_min, y_max = y_min / h, y_max / h
            x_min, x_max = x_min / w, x_max / w
            mask_coords = np.array([x_min, y_min, x_max, y_max])

        sample["mask"] = mask.astype(np.float32)
        sample["mask_coords"] = mask_coords

        return sample


class GenBBox(object):
    def __init__(self, coe_scale=0):
        self.coe_scale = coe_scale

    def __call__(self, sample):
        alpha = sample["alpha"]
        height, width = alpha.shape

        coe = random.uniform(0, self.coe_scale)
        coords = np.nonzero(alpha)

        if coords[0].size == 0 or coords[1].size == 0:
            sample["bbox_mask"], sample["bbox_coords"] = np.zeros_like(alpha).astype(np.float32), np.array([0, 0, 1, 1])
        else:
            binary_mask = alpha > 0
            labeled_array, num_features = label(binary_mask)
            y_min, x_min = np.argwhere(binary_mask).min(axis=0)
            y_max, x_max = np.argwhere(binary_mask).max(axis=0)
            if num_features > 0:
                component_coords = [np.argwhere(labeled_array == i) for i in range(1, num_features + 1)]
                areas = [coords.shape[0] for coords in component_coords]

                sorted_areas_idx = np.argsort(areas)[::-1]
                max_area_idx = sorted_areas_idx[0]
                second_max_area_idx = sorted_areas_idx[1] if len(sorted_areas_idx) > 1 else None

                max_area = areas[max_area_idx]
                second_max_area = areas[second_max_area_idx] if second_max_area_idx is not None else 0

                if max_area >= 10 * second_max_area:
                    max_coords = component_coords[max_area_idx]
                    y_min, x_min = max_coords.min(axis=0)
                    y_max, x_max = max_coords.max(axis=0)

            # Calculate padding_y and padding_x
            padding_y = int(coe * (y_max - y_min))
            padding_x = int(coe * (x_max - x_min))

            # Randomly decide whether to add or subtract padding
            y_min_padding = padding_y if random.choice([True, False]) else -padding_y
            y_max_padding = padding_y if random.choice([True, False]) else -padding_y
            x_min_padding = padding_x if random.choice([True, False]) else -padding_x
            x_max_padding = padding_x if random.choice([True, False]) else -padding_x

            # Apply the padding and ensure it does not exceed the image boundaries
            y_min, y_max = max(0, y_min + y_min_padding), min(height, y_max + y_max_padding)
            x_min, x_max = max(0, x_min + x_min_padding), min(width, x_max + x_max_padding)

            # Generate the bounding box mask
            bbox_mask = np.zeros_like(alpha)
            bbox_mask[y_min:y_max, x_min:x_max] = 1

            y_min, y_max = y_min / height, y_max / height
            x_min, x_max = x_min / width, x_max / width

            # Update the sample dictionary with the bounding box mask and coordinates
            sample["bbox_mask"], sample["bbox_coords"] = bbox_mask.astype(np.float32), np.array([x_min, y_min, x_max, y_max])

        return sample


class GenPoint(object):
    def __init__(self, thres=0, psm="gauss", radius=20):
        self.thres = thres
        self.psm = psm
        self.radius = radius

    def __call__(self, sample):
        alpha = sample["alpha"]
        height, width = alpha.shape
        radius = self.radius

        alpha_mask = (alpha > self.thres).astype(np.float32)
        y_coords, x_coords = np.where(alpha_mask == 1)

        num_points = 10

        if len(y_coords) < num_points:
            sample["point_mask"], sample["point_coords"] = np.zeros_like(alpha).astype(np.float32), np.zeros(20, dtype=np.float32)
            return sample

        selected_indices = np.random.choice(len(y_coords), size=num_points, replace=False)

        point_mask = np.zeros_like(alpha, dtype=np.float32)
        point_coords = []

        for idx in selected_indices:
            y_center = y_coords[idx]
            x_center = x_coords[idx]

            if self.psm == "gauss":
                tmp_mask = np.zeros_like(alpha, dtype=np.float32)
                tmp_mask[y_center, x_center] = 1
                tmp_mask = scipy.ndimage.gaussian_filter(tmp_mask, sigma=radius)
                tmp_mask /= np.max(tmp_mask)
            elif self.psm == "circle":
                tmp_mask = np.zeros_like(alpha, dtype=np.float32)
                for i in range(-radius, radius + 1):
                    for j in range(-radius, radius + 1):
                        if i**2 + j**2 <= radius**2 and 0 <= x_center + i < alpha.shape[0] and 0 <= y_center + j < alpha.shape[1]:
                            tmp_mask[y_center + j, x_center + i] = 1

            point_mask = np.maximum(point_mask, tmp_mask)

            y_norm = y_center / height
            x_norm = x_center / width
            point_coords.append(x_norm)
            point_coords.append(y_norm)
        if len(point_coords) < 20:
            point_coords = np.concatenate([point_coords, np.zeros(20 - len(point_coords))])

        sample["point_mask"] = point_mask.astype(np.float32)
        sample["point_coords"] = np.array(point_coords[:20])

        return sample


class Gen_Add_Mask_Coord(object):
    def __call__(self, sample):
        trimap = sample["trimap"]

        sample["auto_mask"] = np.ones_like(trimap).astype(np.float32)
        sample["auto_coords"] = np.array([0, 0, 1, 1])
        sample["trimap_coords"] = np.array([0, 0, 1, 1])
        return sample


class CutMask(object):
    def __init__(self, perturb_prob=0):
        self.perturb_prob = perturb_prob

    def __call__(self, sample):
        if np.random.rand() < self.perturb_prob:
            return sample

        mask = sample["mask"]  # H x W, trimap 0--255, segmask 0--1, alpha 0--1
        h, w = mask.shape
        perturb_size_h, perturb_size_w = random.randint(h // 4, h // 2), random.randint(w // 4, w // 2)
        x = random.randint(0, h - perturb_size_h)
        y = random.randint(0, w - perturb_size_w)
        x1 = random.randint(0, h - perturb_size_h)
        y1 = random.randint(0, w - perturb_size_w)

        mask[x : x + perturb_size_h, y : y + perturb_size_w] = mask[x1 : x1 + perturb_size_h, y1 : y1 + perturb_size_w]

        sample["mask"] = mask
        return sample


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, alpha = sample["image"], sample["alpha"]

        ### resize
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, self.size, interpolation=cv2.INTER_LINEAR)

        sample["alpha"] = alpha
        sample["image"] = image

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(512, 512)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2

    def __call__(self, sample):
        image, alpha, trimap = (
            sample["image"],
            sample["alpha"],
            sample["trimap"],
        )
        h, w = trimap.shape
        if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
            ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                image = cv2.resize(
                    image,
                    (int(w * ratio), int(h * ratio)),
                    interpolation=cv2.INTER_LINEAR,
                )
                alpha = cv2.resize(
                    alpha,
                    (int(w * ratio), int(h * ratio)),
                    interpolation=cv2.INTER_LINEAR,
                )
                trimap = cv2.resize(trimap, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
                h, w = trimap.shape
        small_trimap = cv2.resize(trimap, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(
            zip(*np.where(small_trimap[self.margin // 4 : (h - self.margin) // 4, self.margin // 4 : (w - self.margin) // 4] == 0.5))
        )
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            left_top = (
                np.random.randint(0, h - self.output_size[0] + 1),
                np.random.randint(0, w - self.output_size[1] + 1),
            )
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0] * 4, unknown_list[idx][1] * 4)

        image_crop = image[
            left_top[0] : left_top[0] + self.output_size[0],
            left_top[1] : left_top[1] + self.output_size[1],
            :,
        ]
        alpha_crop = alpha[left_top[0] : left_top[0] + self.output_size[0], left_top[1] : left_top[1] + self.output_size[1]]
        trimap_crop = trimap[left_top[0] : left_top[0] + self.output_size[0], left_top[1] : left_top[1] + self.output_size[1]]

        if len(np.where(trimap == 0.5)[0]) == 0:
            image_crop = cv2.resize(image, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)

        sample.update({"image": image_crop, "alpha": alpha_crop, "trimap": trimap_crop})
        return sample


class RandomGray(object):
    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, sample):
        image = sample["image"]
        if random.random() < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.stack([image] * 3, axis=-1)
        sample["image"] = image

        return sample


class Normalize:
    """Normalize image values by first mapping from [0, 255] to [0, 1] and then
    applying standardization.
    """

    def normalize_img(self, img):
        assert img.dtype == np.float32
        scaled = img.copy() * 2 - 1
        return scaled

    def __call__(self, sample):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        sample["image"] = self.normalize_img(sample["image"])
        sample["alpha"] = self.normalize_img(sample["alpha"])
        sample["trimap"] = self.normalize_img(sample["trimap"])
        sample["mask"] = self.normalize_img(sample["mask"])
        sample["bbox_mask"] = self.normalize_img(sample["bbox_mask"])
        sample["point_mask"] = self.normalize_img(sample["point_mask"])
        sample["auto_mask"] = self.normalize_img(sample["auto_mask"])
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """

    def __call__(self, sample):
        image, alpha, is_trans = sample["image"], sample["alpha"], sample["is_trans"]

        sample["image"], sample["alpha"], sample["is_trans"] = (
            F.to_tensor(image).float(),
            F.to_tensor(alpha).float(),
            torch.tensor(is_trans).long(),
        )
        if "trimap" in sample and "trimap_coords" in sample:
            sample["trimap"], sample["trimap_coords"] = (
                F.to_tensor(sample["trimap"]).float(),
                torch.from_numpy(sample["trimap_coords"]).float(),
            )
        if "mask" in sample and "mask_coords" in sample:
            sample["mask"], sample["mask_coords"] = (
                F.to_tensor(sample["mask"]).float(),
                torch.from_numpy(sample["mask_coords"]).float(),
            )
        if "bbox_mask" in sample and "bbox_coords" in sample:
            sample["bbox_mask"], sample["bbox_coords"] = (
                F.to_tensor(sample["bbox_mask"]).float(),
                torch.from_numpy(sample["bbox_coords"]).float(),
            )
        if "point_mask" in sample and "point_coords" in sample:
            sample["point_mask"], sample["point_coords"] = (
                F.to_tensor(sample["point_mask"]).float(),
                torch.from_numpy(sample["point_coords"]).float(),
            )
        if "auto_mask" in sample and "auto_coords" in sample:
            sample["auto_mask"], sample["auto_coords"] = (
                F.to_tensor(sample["auto_mask"]).float(),
                torch.from_numpy(sample["auto_coords"]).float(),
            )
        return sample


def get_coco_matting_data(phase):
    if phase == "train":
        json_path = DATA_TRAIN_PATH["COCO_Matting"]
        samples = generate_samples_from_coco_matte(json_path)
    else:
        raise ValueError("This dataset currently does not support the test set.")
    print(f"get {len(samples)} samples from COCO-Matte DataSet")
    return samples


def get_ref_matte_data(phase):
    if phase == "train":
        image_path = "data/txt/RefMatte_10k_image_samples.txt"
        alpha_path = "data/txt/RefMatte_10k_mask_samples.txt"
        samples = generate_samples_from_refmatte_10k(image_path, alpha_path)
    elif phase == "test":
        image_path = os.path.join(DATA_TEST_PATH["RefMatte_RW_100"], DATA_TEST_ARGS["RefMatte_RW_100"][0])
        alpha_path = os.path.join(DATA_TEST_PATH["RefMatte_RW_100"], DATA_TEST_ARGS["RefMatte_RW_100"][1])
        json_path = os.path.join(DATA_TEST_PATH["RefMatte_RW_100"], DATA_TEST_ARGS["RefMatte_RW_100"][2])
        samples = generate_samples_from_refmatte(image_path, alpha_path, json_path)
    print(f"get {len(samples)} samples from RefMatte DataSet")
    return samples


def get_composition_1k_data(phase):
    global BG_LIST
    if phase == "train":
        samples = []
        fg_dir = os.path.join(DATA_TRAIN_PATH["Composition-1K"], DATA_TRAIN_ARGS["Composition-1K"][0])
        label_dir = os.path.join(DATA_TRAIN_PATH["Composition-1K"], DATA_TRAIN_ARGS["Composition-1K"][1])
        sample_num = DATA_TRAIN_ARGS["Composition-1K"][2]

        label_list = os.listdir(label_dir)
        bg_num = len(BG_LIST)
        with open('data/txt/Composition_1k_transparent_samples.txt', "r") as file:
            trans_list = [line.strip() for line in file]
        for label_name in label_list:
            for i in range(sample_num):
                path = {}
                fg_name = label_name
                bg_index = label_list.index(label_name) * sample_num + i
                fg_path = os.path.join(fg_dir, fg_name)
                bg_path = os.path.join(BG_PATH, BG_LIST[bg_index % bg_num])
                label_path = os.path.join(label_dir, label_name)
                path["fg"] = fg_path
                path["bg"] = bg_path
                path["alpha"] = label_path
                path["is_trans"] = int(fg_name in trans_list)
                samples.append(path)
    else:
        raise ValueError("This dataset currently does not support the test set.")
    print(f"get {len(samples)} samples from Composition-1K DataSet")
    return samples


def get_dis_646_data(phase):
    global BG_LIST
    if phase == "train":
        fg_dir = os.path.join(DATA_TRAIN_PATH["DIS-646"], DATA_TRAIN_ARGS["DIS-646"][0])
        label_dir = os.path.join(DATA_TRAIN_PATH["DIS-646"], DATA_TRAIN_ARGS["DIS-646"][1])
        sample_num = DATA_TRAIN_ARGS["DIS-646"][2]

        label_list = os.listdir(label_dir)
        bg_num = len(BG_LIST)
        samples = []
        with open('data/txt/DIS_646_transparent_samples.txt', "r") as file:
            trans_list = [line.strip() for line in file]
        for label_name in label_list:
            for i in range(sample_num):
                path = {}
                fg_name = label_name
                bg_index = label_list.index(label_name) * sample_num + i
                fg_path = os.path.join(fg_dir, fg_name)
                bg_path = os.path.join(BG_PATH, BG_LIST[bg_index % bg_num])
                label_path = os.path.join(label_dir, label_name)
                path["fg"] = fg_path
                path["bg"] = bg_path
                path["alpha"] = label_path
                path["is_trans"] = int(fg_name in trans_list)
                samples.append(path)
    else:
        raise ValueError("This dataset currently does not support the test set.")
    print(f"get {len(samples)} samples from DIS-646 DataSet")
    return samples


def get_p3m_10k_data(phase):
    if phase == "train":
        image_dir = os.path.join(DATA_TRAIN_PATH["P3M-10K"], DATA_TRAIN_ARGS["P3M-10K"][0])
        label_dir = os.path.join(DATA_TRAIN_PATH["P3M-10K"], DATA_TRAIN_ARGS["P3M-10K"][1])

        label_list = os.listdir(label_dir)
        samples = []

        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["is_trans"] = 0
            samples.append(path)
    elif phase == "test":
        image_dir = os.path.join(DATA_TEST_PATH["P3M-500-NP"], DATA_TEST_ARGS["P3M-500-NP"][0])
        label_dir = os.path.join(DATA_TEST_PATH["P3M-500-NP"], DATA_TEST_ARGS["P3M-500-NP"][1])

        label_list = os.listdir(label_dir)
        samples = []

        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["is_trans"] = 0
            samples.append(path)
    print(f"get {len(samples)} samples from P3M-10K DataSet")
    return samples


def get_am_2k_data(phase):
    global BG_LIST
    if phase == "train":
        samples = []
        image_dir = os.path.join(DATA_TRAIN_PATH["AM-2K"], DATA_TRAIN_ARGS["AM-2K"][0])
        label_dir = os.path.join(DATA_TRAIN_PATH["AM-2K"], DATA_TRAIN_ARGS["AM-2K"][1])
        fg_dir = os.path.join(DATA_TRAIN_PATH["AM-2K"], DATA_TRAIN_ARGS["AM-2K"][2])
        sample_num = DATA_TRAIN_ARGS["AM-2K"][3]

        label_list = os.listdir(label_dir)
        bg_num = len(BG_LIST)

        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["is_trans"] = 0
            samples.append(path)

        for label_name in label_list:
            for i in range(sample_num):
                path = {}
                fg_name = label_name
                bg_index = label_list.index(label_name) * sample_num + i
                fg_path = os.path.join(fg_dir, fg_name)
                bg_path = os.path.join(BG_PATH, BG_LIST[bg_index % bg_num])
                label_path = os.path.join(label_dir, label_name)
                path["fg"] = fg_path
                path["bg"] = bg_path
                path["alpha"] = label_path
                path["is_trans"] = 0
                samples.append(path)
    elif phase == "test":
        samples = []
        image_dir = os.path.join(DATA_TEST_PATH["AM-2K"], DATA_TEST_ARGS["AM-2K"][0])
        label_dir = os.path.join(DATA_TEST_PATH["AM-2K"], DATA_TEST_ARGS["AM-2K"][1])
        trimap_dir = os.path.join(DATA_TEST_PATH["AM-2K"], DATA_TEST_ARGS["AM-2K"][2])

        label_list = os.listdir(label_dir)

        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            trimap_path = os.path.join(trimap_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["trimap"] = trimap_path
            path["is_trans"] = 0
            samples.append(path)
    print(f"get {len(samples)} samples from AM-2K DataSet")
    return samples


def get_rwp_636_data(phase):
    if phase == "train":
        samples = []
        image_dir = os.path.join(DATA_TRAIN_PATH["RWP-636"], DATA_TRAIN_ARGS["RWP-636"][0])
        label_dir = os.path.join(DATA_TRAIN_PATH["RWP-636"], DATA_TRAIN_ARGS["RWP-636"][1])
        label_list = os.listdir(label_dir)

        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["is_trans"] = 0
            samples.append(path)
    else:
        raise ValueError("This dataset currently does not support the test set.")
    print(f"get {len(samples)} samples from RWP-636 DataSet")
    return samples


def get_trans_460_data(phase):
    global BG_LIST
    if phase == "train":
        samples = []
        image_dir = os.path.join(DATA_TRAIN_PATH["Trans-460"], DATA_TRAIN_ARGS["Trans-460"][0])
        label_dir = os.path.join(DATA_TRAIN_PATH["Trans-460"], DATA_TRAIN_ARGS["Trans-460"][1])
        sample_num = DATA_TRAIN_ARGS["Trans-460"][2]
        label_list = os.listdir(label_dir)
        bg_num = len(BG_LIST)

        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["is_trans"] = 1
            samples.append(path)

        for label_name in label_list:
            for i in range(sample_num):
                path = {}
                fg_name = os.path.splitext(label_name)[0] + ".jpg"
                bg_index = label_list.index(label_name) * sample_num + i
                fg_path = os.path.join(image_dir, fg_name)
                bg_path = os.path.join(BG_PATH, BG_LIST[bg_index % bg_num])
                label_path = os.path.join(label_dir, label_name)
                path["fg"] = fg_path
                path["bg"] = bg_path
                path["alpha"] = label_path
                path["is_trans"] = 1
                samples.append(path)
    elif phase == "test":
        samples = []
        image_dir = os.path.join(DATA_TEST_PATH["Trans-460"], DATA_TEST_ARGS["Trans-460"][0])
        label_dir = os.path.join(DATA_TEST_PATH["Trans-460"], DATA_TEST_ARGS["Trans-460"][1])
        trimap_dir = os.path.join(DATA_TEST_PATH["Trans-460"], DATA_TEST_ARGS["Trans-460"][2])
        label_list = os.listdir(label_dir)

        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            trimap_path = os.path.join(trimap_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["trimap"] = trimap_path
            path["is_trans"] = 1
            samples.append(path)
    print(f"get {len(samples)} samples from Trans-460 DataSet")
    return samples


def get_aim_500_data(phase):
    if phase == "test":
        samples = []
        image_dir = os.path.join(DATA_TEST_PATH["AIM-500"], DATA_TEST_ARGS["AIM-500"][0])
        label_dir = os.path.join(DATA_TEST_PATH["AIM-500"], DATA_TEST_ARGS["AIM-500"][1])
        trimap_dir = os.path.join(DATA_TEST_PATH["AIM-500"], DATA_TEST_ARGS["AIM-500"][2])
        label_list = os.listdir(label_dir)
        with open('data/txt/AIM_500_transparent_samples.txt', "r") as file:
            trans_list = [line.strip() for line in file]
        
        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            trimap_path = os.path.join(trimap_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["trimap"] = trimap_path
            path["is_trans"] = int(image_name in trans_list)
            samples.append(path)
    else:
        raise ValueError("This dataset currently does not support the train set.")
    print(f"get {len(samples)} samples from AIM-500 DataSet")
    return samples


def get_uhrsd_data(phase):
    if phase == "train":
        image_dir = os.path.join(DATA_TRAIN_PATH["UHRSD"], DATA_TRAIN_ARGS["UHRSD"][0])
        label_dir = os.path.join(DATA_TRAIN_PATH["UHRSD"], DATA_TRAIN_ARGS["UHRSD"][1])

        label_list = os.listdir(label_dir)
        samples = []

        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["is_trans"] = 0
            samples.append(path)

        image_dir = os.path.join(DATA_TEST_PATH["UHRSD"], DATA_TEST_ARGS["UHRSD"][0])
        label_dir = os.path.join(DATA_TEST_PATH["UHRSD"], DATA_TEST_ARGS["UHRSD"][1])

        label_list = os.listdir(label_dir)

        for label_name in label_list:
            path = {}
            image_name = os.path.splitext(label_name)[0] + ".jpg"
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, label_name)
            path["image"] = image_path
            path["alpha"] = label_path
            path["is_trans"] = 0
            samples.append(path)
    print(f"get {len(samples)} samples from UHRSD DataSet")
    return samples


def get_train_data(setnames):
    samples = []
    if "Composition-1K" in setnames:
        samples += get_composition_1k_data("train")
    if "DIS-646" in setnames:
        samples += get_dis_646_data("train")
    if "P3M-10K" in setnames:
        samples += get_p3m_10k_data("train")
    if "AM-2K" in setnames:
        samples += get_am_2k_data("train")
    if "RWP-636" in setnames:
        samples += get_rwp_636_data("train")
    if "Trans-460" in setnames:
        samples += get_trans_460_data("train")
    if "COCO-Matting" in setnames:
        samples += get_coco_matting_data("train")
    if "UHRSD" in setnames:
        samples += get_uhrsd_data("train")
    if "RefMatte" in setnames:
        samples += get_ref_matte_data("train")

    print(f"get total {len(samples)} train data")

    return samples


def get_test_data(setname):
    if setname == "P3M-500-NP":
        samples = get_p3m_10k_data("test")
    elif setname == "AM-2K":
        samples = get_am_2k_data("test")
    elif setname == "AIM-500":
        samples = get_aim_500_data("test")
    elif setname == "Trans-460":
        samples = get_trans_460_data("test")
    elif setname == "RefMatte_RW_100":
        samples = get_ref_matte_data("test")

    print(f"get total {len(samples)} test data")

    return samples


class DataGenerator(Dataset):
    def __init__(self, set_list, phase="train", crop_size=512, psm="gauss", radius=20):
        self.phase = phase
        if phase == "train":
            self.samples = get_train_data(set_list)
        else:
            self.samples = get_test_data(set_list)

        train_trans = [
            Resize((crop_size, crop_size)),
            GenTrimap(),
            GenMask(),
            GenBBox(0.01),
            GenPoint(0.8, psm, radius),
            Gen_Add_Mask_Coord(),
            RandomGray(0.2),
            Normalize(),
            ToTensor(),
        ]

        test_trans = [
            Resize((1024, 1024)),
            GenTrimap(),
            GenMask(),
            GenBBox(),
            GenPoint(0.8, psm, radius + 10),
            Gen_Add_Mask_Coord(),
            Normalize(),
            ToTensor(),
        ]

        self.transform = {
            "train": transforms.Compose(train_trans),
            "test": transforms.Compose(test_trans),
        }[phase]

        self.fg_num = len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx % self.fg_num]
        alpha = cv2.imread(data["alpha"], 0).astype(np.float32) / 255.0
        H, W = alpha.shape
        image_name = os.path.split(data["alpha"])[-1]

        if "image" in data:
            image = cv2.imread(data["image"], 1).astype(np.float32) / 255.0
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            bg = cv2.imread(data["bg"], 1)
            fg = cv2.imread(data["fg"], 1)
            image, alpha = composition(fg, bg, alpha)
            image = image.astype(np.float32) / 255.0
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if "trimap" in data:
            trimap = cv2.imread(data["trimap"], 0).astype(np.float32)
            trimap[trimap == 128] = 0.5
            trimap[trimap == 255] = 1.0
        else:
            trimap = None

        if "mask" in data:
            mask = cv2.imread(data["mask"], 0).astype(np.float32)
            mask[mask == 255] = 1.0
        else:
            mask = None
        if "caption" in data:
            caption = data["caption"]
        else:
            caption = ""
        if "is_trans" in data:
            is_trans = data["is_trans"]
        else:
            raise ValueError("There is no is_trans in the sample.")
        sample = {
            "image": image,
            "alpha": alpha,
            "trimap": trimap,
            "mask": mask,
            "is_trans": is_trans,
            "image_name": image_name,
            "caption": caption,
            "hw": (H, W),
        }

        assert alpha.shape == image.shape[:2]
        assert image.shape[-1] == 3

        sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.fg_num


def resize_bg(bg, alpha):
    if bg.shape[0] > alpha.shape[0] and bg.shape[1] > alpha.shape[1]:
        random_h = random.randint(0, bg.shape[0] - alpha.shape[0])
        random_w = random.randint(0, bg.shape[1] - alpha.shape[1])
        bg = bg[random_h : random_h + alpha.shape[0], random_w : random_w + alpha.shape[1], :]
    else:
        bg = cv2.resize(bg, (alpha.shape[1], alpha.shape[0]), cv2.INTER_LINEAR)
    return bg


def composition(fg, bg, alpha):
    ori_alpha = alpha.copy()
    h, w = alpha.shape
    if random.random() < 1:
        bg = cv2.resize(bg, (2 * w, h), cv2.INTER_LINEAR)
        alpha = np.concatenate((alpha, alpha), axis=1)
        fg = np.concatenate((fg, fg), axis=1)
    else:
        bg = cv2.resize(bg, (w, h), cv2.INTER_LINEAR)
    alpha[alpha < 0] = 0
    alpha[alpha > 1] = 1
    fg[fg < 0] = 0
    fg[fg > 255] = 255
    bg[bg < 0] = 0
    bg[bg > 255] = 255
    if random.random() < 0.5:
        rand_kernel = random.choice([20, 30, 40, 50, 60])
        bg = cv2.blur(bg, (rand_kernel, rand_kernel))
    image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
    if ori_alpha.shape != alpha.shape:
        if random.random() < 0.5:
            alpha[:, :w] = 0
        else:
            alpha[:, w:] = 0
    return image, alpha
