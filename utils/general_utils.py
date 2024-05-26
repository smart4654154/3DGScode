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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0#是(height, width, channels)
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)#通道，高度，宽度
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    """
    创建一个学习率调度函数，该函数根据训练进度动态调整学习率

    :param lr_init: 初始学习率。
    :param lr_final: 最终学习率。
    :param lr_delay_steps: 学习率延迟步数，在这些步数内学习率将被降低。
    :param lr_delay_mult: 学习率延迟乘数，用于计算初始延迟学习率。
    :param max_steps: 最大步数，用于规范化训练进度。
    :return: 一个函数，根据当前步数返回调整后的学习率。
    """
    def helper(step):
        # 如果步数小于0或学习率为0，直接返回0，表示不进行优化
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        # 如果设置了学习率延迟步数，计算延迟调整后的学习率
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        # 根据步数计算学习率的对数线性插值，实现从初始学习率到最终学习率的平滑过渡
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        # 返回调整后的学习率
        return delay_rate * log_lerp

    return helper
#当你调用get_expon_lr_func并传入参数后，你会得到一个返回学习率的函数，可以用于训练过程中的学习率调度。

def strip_lowerdiag(L):
    """
    取下协方差矩阵的下三角部分。
    从协方差矩阵中提取六个独立参数。

    :param L: 协方差矩阵。
    :return: 六个独立参数组成的张量。
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    # 提取协方差矩阵的独立元素
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    """
    提取协方差矩阵的对称部分。

    :param sym: 协方差矩阵。
    :return: 对称部分。
    """
    return strip_lowerdiag(sym)

def build_rotation(r):
    # 将 N x 4 的旋转四元组转换成 N x 3 x 3 的旋转矩阵, N 为点云中点的个数或者3D gaussian的个数
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    """
    构建3D高斯模型的尺度-旋转矩阵。

    :param s: 尺度参数。
    :param r: 旋转参数。
    :return: 尺度-旋转矩阵。
    """
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")  # 初始化尺度矩阵
    R = build_rotation(r)  # 计算旋转矩阵

    # 设置尺度矩阵的对角线元素
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L  # 应用旋转
    return L

def safe_state(silent):#控制输出
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                    #该函数将字符串x中的换行符"\n"替换为" [日期时间]\n"的形式，并将结果写入到old_f文件中
                    #datetime.now() 返回当前的日期和时间对象，而 strftime() 是一个用于格式化日期和时间的方法
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

# 这段代码定义了一个函数 safe_state(silent)，该函数的作用是在执行期间重定向标准输出（sys.stdout）
# 到一个新的类 F 的实例。这个类 F 在写入时会检查是否需要在每行结尾处添加时间戳，以及是否需要替换换行符。
#
# 具体来说，函数的实现步骤如下：
#
# 将原始的标准输出保存在 old_f 变量中。
# 定义一个名为 F 的新类，该类具有以下方法：
# __init__(self, silent)：初始化方法，接受一个参数 silent。
# write(self, x)：写入方法，检查 silent 属性，如果不是静默模式，则在每行结尾添加当前时间戳，并将文本写入原始标准输出。
# flush(self)：刷新方法，将原始标准输出的缓冲区刷新。
# 创建 F 类的实例并将其赋值给 sys.stdout，从而重定向标准输出到新的类实例。
# 设置随机种子以确保结果的可重复性。
# 最后，将 PyTorch 的随机种子设置为 0，并将当前 CUDA 设备设置为 "cuda:0"（如果可用的话）。
