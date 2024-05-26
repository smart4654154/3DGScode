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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False






def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0

    tb_writer = prepare_output_and_logger(dataset)
    # 初始化高斯模型，用于表示场景中的每个点的3D高斯分布
    gaussians = GaussianModel(dataset.sh_degree)
    # 初始化场景对象，加载数据集和对应的相机参数
    scene = Scene(dataset, gaussians)
    # 为高斯模型参数设置优化器和学习率调度器
    gaussians.training_setup(opt)
    # 如果提供了checkpoint，则从checkpoint加载模型参数并恢复训练进度
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # 设置背景颜色，白色或黑色取决于数据集要求
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing=True)#启用了时间测量功能
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # 使用tqdm库创建进度条，追踪训练进度
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # 记录迭代开始时间
        iter_start.record()
#用于记录迭代开始的时间点。它被用来跟踪和计算模型训练或数据处理过程中的时间消耗。
        # 在训练循环开始时调用此方法，可以将其与iter_end.record()方法结合使用，以计算每次迭代的时间。
        # 根据当前迭代次数更新学习率
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代，提升球谐函数的次数以改进模型复杂度
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个训练用的相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        #首先，它会检查视点栈是否为空。如果为空，则通过调用scene.getTrainCameras().copy()将场景中的训练相机复制到视点栈中。
# 然后，它使用randint(0, len(viewpoint_stack)-1)生成一个随机索引，用于从视点栈中选择一个相机。
# 最后，它使用pop()方法从视点栈中移除选定的相机，并将其赋值给viewpoint_cam变量。

        # 如果达到调试起始点，启用调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 根据设置决定是否使用随机背景颜色
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 渲染当前视角的图像
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 计算渲染图像与真实图像之间的损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        # 记录迭代结束时间
        iter_end.record()

        with torch.no_grad():
            # 更新进度条和损失显示
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)#进度条的进度增加10个单位
            if iteration == opt.iterations:
                progress_bar.close()
            # 使用torch.no_grad()
            # 上下文管理器，可以禁用梯度计算，以减少内存和计算时间。
            # 通过ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log公式，更新累积平均损失值ema_loss_for_log，其中0
            # .4
            # 和0
            # .6
            # 是权重系数。
            # 若当前迭代次数iteration能被10整除，将累积平均损失值ema_loss_for_log格式化后，通过progress_bar.set_postfix()
            # 方法设置到进度条的后缀显示中，并通过progress_bar.update()
            # 方法更新进度条。
            # 若当前迭代次数iteration等于预设的最大迭代次数opt.iterations，则通过progress_bar.close()
            # 方法关闭进度条。
            # 定期记录训练数据并保存模型
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 在指定迭代区间内，对3D高斯模型进行增密和修剪
            if iteration < opt.densify_until_iter:#修剪的上限次数
                # gaussians.max_radii2D[取出可见的，再赋值]
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)#添加密集化状态

                if 1:
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:#大于开始的循环500，%密度间隔100==0
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None#>不透明度重置区间
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)#密度和修剪

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()#不透明区间==0或（白色+密度_开始_iter）

            # 执行优化器的一步，并准备下一次迭代
            if iteration < opt.iterations:#opt.iterations是训练上限
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # 定期保存checkpoint
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")











def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        #该函数用于创建一个SummaryWriter对象
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:  # 将 L1 loss、总体 loss 和迭代时间写入 TensorBoard。
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 在指定的测试迭代次数，进行渲染并计算 L1 loss 和 PSNR。
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})#创建一个包含余数的列表。

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # 获取渲染结果和真实图像
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    # 返回一个包含渲染结果的字典。
                    # image是渲染结果中
                    # "render"
                    # 键对应的值，它是一个张量，其像素值会被限制在0.0到1之间。
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):  # 在 TensorBoard 中记录渲染结果和真实图像
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                            #是TensorFlow中用于在TensorBoard的图像面板上添加多张图像的函数
                    # 计算 L1 loss 和 PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                # 计算平均 L1 loss 和 PSNR
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                # 在控制台打印评估结果
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                # 在 TensorBoard 中记录评估结果
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # 在 TensorBoard 中记录场景的不透明度直方图和总点数。
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()  # 使用 torch.cuda.empty_cache() 清理 GPU 内存。

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    #解析命令行参数
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    #这段代码的目的是为了在执行过程中控制标准输出的行为，添加时间戳并在需要时禁止输出，以便在某些场景下更方便地进行调试和记录。
    #（应该就是输出一些系统的状态的）
    safe_state(args.quiet)


    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port) #这行代码初始化一个 GUI 服务器，使用 args.ip 和 args.port 作为参数。这可能是一个用于监视和控制训练过程的图形用户界面的一部分。
    torch.autograd.set_detect_anomaly(args.detect_anomaly) #这行代码设置 PyTorch 是否要检测梯度计算中的异常。
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # 输入的参数包括：lp.extract(args)模型的参数（数据集的位置）、优化器的参数、其他pipeline的参数，测试迭代次数、保存迭代次数 、检查点迭代次数 、开始检查点 、调试起点
    # All done
    print("\nTraining complete.")
