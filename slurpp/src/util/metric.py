# Author: Bingxin Ke
# Last modified: 2024-02-15


import pandas as pd
import torch
import numpy as np 
import lpips


# Adapted from: https://github.com/victoresque/pytorch-template/blob/master/utils/util.py
class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()

def mae_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_diff = torch.abs(diff)
    mae = torch.sum(abs_diff, (-1, -2)) / n
    return mae.mean()


def rmse_linear(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss

lpips_fn = lpips.LPIPS(net='alex').to("cuda")


def psnr(output, target, valid_mask=None):
    return peak_signal_noise_ratio(target, output)

def ssim(output, target, valid_mask=None):
    return structural_similarity(target, output)

def lpips_loss(output, target, valid_mask=None):
    return lpips_fn(target * 2.0 -1.0, output * 2.0 - 1.0)

def peak_signal_noise_ratio(image_true, image_test, dmin=0, dmax=1):
    """ Assert image in [0,1] """
    assert image_test.shape == image_true.shape
    assert image_test.dtype == image_true.dtype

    true_min, true_max = image_true.min(), image_true.max()
    if true_max > dmax or true_min < dmin:
        raise ValueError(
            "im_true has intensity values outside the range expected for "
            "its data type.  Please manually specify the data_range")
    if true_min >= 0:
        data_range = dmax
    else:
        data_range = dmax - dmin

    err = (image_true - image_test)**2
    if isinstance(image_true, np.ndarray):
        psnr = 10 * np.log10((data_range ** 2) / err.mean())
    else:
        psnr = 10 * torch.log10((data_range ** 2) / err.mean())
    return psnr

def rgb2qgray(img: torch.Tensor):
    """ Input float tensor with shape b,3,h,w and range [0,1] 
        Return uint8 tensor with shape b,1,h,w and range [0,255] """
    b, _, h, w = img.shape
    img = img.reshape((3, h * w))
    coefficient = torch.tensor(
        [[0.2989, 0.5870, 0.1140]], dtype=img.dtype, device=img.device)
    img = coefficient @ img
    img = img.reshape((b, 1, h, w))
    img = (img * 255
           ).type(torch.uint8)
    return img


def structural_similarity(im1, im2, K1=0.01, K2=0.03, win_size=7, R=255):
    im1, im2 = (rgb2qgray(x).type(torch.float32) for x in (im1, im2))
    numel = win_size**2
    kernel = torch.ones((1, 1, win_size, win_size), dtype=im1.dtype,
                        device=im1.device) / numel
    cov_norm = numel / (numel - 1)

    ux, uy = (torch.conv2d(x, kernel) for x in (im1, im2))
    uxx, uyy = (torch.conv2d(x**2, kernel) for x in (im1, im2))
    uxy = torch.conv2d(im1 * im2, kernel)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D
    ssim = S.mean()
    return ssim

