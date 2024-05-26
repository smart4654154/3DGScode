#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    #参数window_size表示窗口的大小，sigma表示高斯函数的标准差
    gauss = torch.Tensor([exp(   -(x - window_size // 2)**2   /   float(2 * sigma ** 2)   ) for x in range(window_size)])
    return gauss / gauss.sum()#归一化，//返回 不大于直接除结果的最大整数

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 首先，通过gaussian(window_size, 1.5)
    # 函数生成一个一维高斯函数，其中window_size表示窗口的大小，1.5
    # 表示高斯函数的标准差。
    # 然后，使用.unsqueeze(1)
    # 将一维高斯函数在第1维上扩展为一个二维张量。
    # 接着，使用.mm(_1D_window.t())
    # 将一维高斯函数进行矩阵乘法，得到一个二维的高斯窗口函数。
    # 最后，使用.unsqueeze(0).unsqueeze(0)
    # 分别扩展了一个维度，将二维高斯窗口函数转换为了一个四维张量。
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    #自动梯度 Variable
    # expand(channel, 1, window_size,
    #        window_size)：
    #        将_2D_window张量在第一个维度上扩展了channel个数，
    #        同时在第二个维度上扩展了1个数，其他两个维度分别扩展了window_size个数。
    #        扩展后的张量大小为(
    #     channel, 1, window_size, window_size)。
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
# expand()函数可以将张量广播到新的形状，但是切记以下两点：
# 只能对维度值为1的维度进行扩展，且扩展的Tensor不会分配新的内存，只是原来的基础上创建新的视图并返回；
# expand(想要的shape，不想改的就设为-1)
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

