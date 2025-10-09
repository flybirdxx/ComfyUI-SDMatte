import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        loss = self.bce_loss(logits, targets)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        intersection = (logits * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (logits.sum() + targets.sum() + self.smooth)
        return 1 - dice


class DistillationCriterion(nn.Module):
    def __init__(self, losses=None):
        super(DistillationCriterion, self).__init__()
        self.losses = losses

    def forward(self, s_fp=None, t_fp=None):
        if "mse_loss" in self.losses:
            losses = sum(F.mse_loss(a, b, reduction="mean") for a, b in zip(s_fp, t_fp))
        if "l1_loss" in self.losses:
            losses += sum(F.l1_loss(a, b) for a, b in zip(s_fp, t_fp))
        return losses * 0.01


class MattingCriterion(nn.Module):
    def __init__(self, losses=None, size=512):
        super(MattingCriterion, self).__init__()
        self.losses = losses
        self.size = size

    def mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def loss_gradient_penalty(self, sample_map, preds, targets):
        preds = preds
        targets = targets

        # sample_map for unknown area
        scale = sample_map.shape[0] * self.size * self.size / torch.sum(sample_map)

        # gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        # gradient in y
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        # loss
        loss = (
            F.l1_loss(delta_pred_x * sample_map, delta_gt_x * sample_map) * scale
            + F.l1_loss(delta_pred_y * sample_map, delta_gt_y * sample_map) * scale
            + 0.01 * torch.mean(torch.abs(delta_pred_x * sample_map)) * scale
            + 0.01 * torch.mean(torch.abs(delta_pred_y * sample_map)) * scale
        )

        return dict(loss_gradient_penalty=loss)

    def lap_loss(self, preds, targets):
        loss = laplacian_loss(preds, targets)

        return dict(lap_loss=loss)

    def unknown_lap_loss(self, sample_map, preds, targets):
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            scale = sample_map.shape[0] * self.size * self.size / torch.sum(sample_map)

        loss = laplacian_loss(preds * sample_map, targets * sample_map) * scale
        return dict(unknown_lap_loss=loss)

    def known_lap_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map == 0] = 1

        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0] * self.size * self.size / torch.sum(new_sample_map)

        loss = laplacian_loss(preds * new_sample_map, targets * new_sample_map) * scale
        return dict(known_lap_loss=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        if torch.sum(sample_map) == 0:
            scale = 0
        else:
            scale = sample_map.shape[0] * self.size * self.size / torch.sum(sample_map)

        loss = F.l1_loss(preds * sample_map, targets * sample_map) * scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, preds, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map == 0] = 1

        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0] * self.size * self.size / torch.sum(new_sample_map)

        loss = F.l1_loss(preds * new_sample_map, targets * new_sample_map) * scale

        if torch.isnan(loss):
            raise ValueError("The computed loss is NaN. Check the input values or computation.")

        return dict(known_l1_loss=loss)

    def l1_loss(self, preds, targets):

        loss = F.l1_loss(preds, targets)
        return dict(l1_loss=loss)

    def mse_loss(self, preds, targets):
        loss = F.mse_loss(preds.float(), targets.float(), reduction="mean")
        return dict(mse_loss=loss)

    def dice_loss(self, pred, target, smooth=1e-5):
        target = (target > 0.5).to(torch.uint8)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Compute the intersection and union
        intersection = (pred_flat * target_flat).sum()  # sum of element-wise multiplication
        union = pred_flat.sum() + target_flat.sum()  # sum of all elements in pred and target

        # Compute the Dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)

        loss = 1 - dice
        return dict(dice_loss=loss)

    def bce_loss(self, probs, target):
        loss_fn = nn.BCELoss()
        target = (target > 0.5).to(torch.uint8)
        loss = loss_fn(probs, target)
        return dict(bce_loss=loss)

    def forward(self, pred=None, label=None, trimap=None):
        if trimap is not None:
            sample_map = torch.zeros_like(trimap)
            sample_map[trimap == 0.0] = 1
        losses = dict()
        for k in self.losses:
            if k in ["unknown_l1_loss", "known_l1_loss", "unknown_lap_loss", "known_lap_loss", "loss_gradient_penalty"]:
                losses.update(getattr(self, k)(sample_map, pred, label))
            else:
                losses.update(getattr(self, k)(pred, label))
        return losses


# -----------------Laplacian Loss-------------------------#
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2**level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels


def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid


def gauss_kernel(device="cpu", dtype=torch.float32):
    kernel = torch.tensor(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], device=device, dtype=dtype
    )
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel


def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode="reflect")
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img


def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img


def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out


def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]
