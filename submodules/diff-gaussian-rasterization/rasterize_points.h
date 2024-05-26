/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 看完525
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
//std::tuple是C++标准库中的一个模板类，它允许你将不同类型的值封装到一个单一的复合对象中。tuple可以存储零个或多个元素，
//每个元素可以是任何类型，它们在元组中按照位置进行区分，而不是名称。
//表示存在一个这样的数据结构，可以用来存储一个整数和五个torch::Tensor对象
//，torch::Tensor就是通过::来访问torch命名空间下的Tensor类。torch通常是PyTorch库的一部分，
//而::用于明确指出我们指的是这个库中的Tensor类型。
//
RasterizeGaussiansCUDA(
	const torch::Tensor& background,//背景
	const torch::Tensor& means3D,//#高斯分布的二维坐标（屏幕空间坐标）。
    const torch::Tensor& colors,//颜色
    const torch::Tensor& opacity,//不透明度
	const torch::Tensor& scales,//缩放
	const torch::Tensor& rotations,// 旋转
	const float scale_modifier,   // 缩放修正
	const torch::Tensor& cov3D_precomp,//预先计算协方差矩阵
	const torch::Tensor& viewmatrix,//视图矩阵 viewmatrix
	const torch::Tensor& projmatrix,//投影矩阵 projmatrix
	const float tan_fovx, // 水平视角
	const float tan_fovy,// 垂直视角
    const int image_height,//
    const int image_width,
	const torch::Tensor& sh,//球谐函数
	const int degree,
	const torch::Tensor& campos,//相机坐标
	const bool prefiltered,//预先计算的过滤器
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);