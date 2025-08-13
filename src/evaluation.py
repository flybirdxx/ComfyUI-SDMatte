import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from detectron2.evaluation import DatasetEvaluator
import numpy as np
import torch
from utils import compute_mse_loss, compute_sad_loss, compute_mad_loss, compute_gradient_loss, compute_connectivity_error
from data import DATA_TEST_ARGS, DATA_TEST_PATH


class MattingEvaluator(DatasetEvaluator):
    def __init__(self):
        self.pred = []
        self.gt = []

    def reset(self):
        self.pred = []
        self.gt = []

    def process(self, input, output):
        """
        collect model_output and gt
        """
        H, W = input["hw"][0].item(), input["hw"][1].item()
        gt = (input["alpha"] + 1.0) / 2.0
        gt = gt.flatten(0, 2).detach().cpu().numpy() * 255
        pred = output.flatten(0, 2).detach().cpu().numpy() * 255
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        self.pred.append(pred)
        self.gt.append(gt)

    def evaluate(self):
        mse_loss = []
        mad_loss = []
        sad_loss = []
        # grad_loss = []
        # conn_loss = []
        for idx in tqdm(range(len(self.gt))):
            label = self.gt[idx].astype(np.float32)
            pred = self.pred[idx].astype(np.float32)

            if pred.shape != label.shape:
                pred = cv2.resize(pred, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)

            # calculate loss
            mse_loss_ = compute_mse_loss(pred, label)
            mad_loss_ = compute_mad_loss(pred, label)
            sad_loss_ = compute_sad_loss(pred, label)
            # grad_loss_ = compute_gradient_loss(pred, label)
            # conn_loss_ = compute_connectivity_error(pred, label)

            # save for average
            mse_loss.append(mse_loss_)
            mad_loss.append(mad_loss_)
            sad_loss.append(sad_loss_)
            # grad_loss.append(grad_loss_)
            # conn_loss.append(conn_loss_)

        mse = np.array(mse_loss).mean() * 1000
        mad = np.array(mad_loss).mean() * 1000
        sad = np.array(sad_loss).mean()
        # grad = np.array(grad_loss).mean()
        # conn = np.array(conn_loss).mean()
        results = {
            "MSE": mse,
            "MAD": mad,
            "SAD": sad,
            # "Grad": grad,
            # "Conn": conn,
        }
        return results


def evaluate(args):
    mse_loss = []
    mad_loss = []
    sad_loss = []
    grad_loss = []
    conn_loss = []

    label_dir = os.path.join(DATA_TEST_PATH[args.setname], DATA_TEST_ARGS[args.setname][1])
    name_list = os.listdir(label_dir)
    if os.path.exists(args.result_path):
        with open(args.result_path, "r+") as f:
            if f.read().strip():  # 检查文件是否非空
                f.seek(0)  # 回到文件开头
                f.truncate()
    for name in tqdm(name_list):
        alpha_path = os.path.join(label_dir, name)
        pred_path = os.path.join(args.pred_dir, name)
        label = cv2.imread(alpha_path, 0).astype(np.float32)
        pred = cv2.imread(pred_path, 0).astype(np.float32)
        if pred.shape != label.shape:
            pred = cv2.resize(pred, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)

        # calculate loss
        mse_loss_ = compute_mse_loss(pred, label) * 1000
        mad_loss_ = compute_mad_loss(pred, label) * 1000
        sad_loss_ = compute_sad_loss(pred, label)
        grad_loss_ = compute_gradient_loss(pred, label)
        conn_loss_ = compute_connectivity_error(pred, label)

        # save for average
        mse_loss.append(mse_loss_)  # mean l2 loss per unknown pixel
        mad_loss.append(mad_loss_)  # l1 loss on unknown area
        sad_loss.append(sad_loss_)
        grad_loss.append(grad_loss_)
        conn_loss.append(conn_loss_)
        results_line = (
            f"{name}: MSE: {mse_loss_:.2f}  MAD: {mad_loss_:.2f}  SAD: {sad_loss_:.2f} Grad: {grad_loss_:.2f} Conn: {conn_loss_:.2f}"
        )
        with open(args.result_path, "a") as f:
            f.write(f"\n{results_line}")

    mse = np.array(mse_loss).mean()
    mad = np.array(mad_loss).mean()
    sad = np.array(sad_loss).mean()
    grad = np.array(grad_loss).mean()
    conn = np.array(conn_loss).mean()

    print(f"MSE: {mse:.2f}  MAD: {mad:.2f}  SAD: {sad:.2f} Grad: {grad:.2f} Conn: {conn:.2f}")
    # print(f"MSE: {mse:.2f}  MAD: {mad:.2f} SAD: {sad:.2f}")
    with open(args.result_path, "a") as f:
        f.write(f"\n{args.setname}: MSE: {mse:.2f}  MAD: {mad:.2f} SAD: {sad:.2f} Grad: {grad:.2f} Conn: {conn:.2f}")
    with open(args.result_path, "r") as f:
        lines = f.readlines()
    sorted_lines = sorted(
        lines,
        key=lambda line: (
            0 if args.setname in line else 1,
            -float(line.split("MSE:")[1].split()[0]) if "MSE:" in line else float("-inf"),
        ),
    )
    with open(args.result_path, "w") as f:
        f.writelines(sorted_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=str, default="infer_output/", required=True, help="output dir")
    parser.add_argument("--setname", type=str, default="dataset/test/alphas/", help="testset name")
    parser.add_argument("--result-path", type=str, help="save the evl result")
    args = parser.parse_args()

    evaluate(args)
