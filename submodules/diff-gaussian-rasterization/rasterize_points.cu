/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
// 这段代码使用了C++中的std::function和Lambda表达式，
//结合了PyTorch库的torch::Tensor类，目的是创建一个std::function对象，该对象可以接受一个size_t类型的参数，
//并将一个torch::Tensor对象的大小调整为指定大小。然后，它返回一个指向调整后数据缓冲区的指针。
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
//std当前缀时，这意味着接下来使用的函数、类或对象是
//定义在 std 命名空间中的标准库部分。
//std::tuple 是C++标准库中的一个模板类，它用于创建能存储不同类型值的固定大小的容器。
//元组（Tuple）允许你组合不同类型的元素到一个单一的对象中
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
     //means3D是 每个3D gaussian的XYZ均值 要求是二维，第二个维度=3
  
  const int P = means3D.size(0);//点数
  const int H = image_height;//高
  const int W = image_width;//宽

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);
//设置means3D选项的数据类型为int32，并返回设置后的选项对象。
  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  // (3, H, W), 在指定的视角下, 对所有3D gaussian进行投影和渲染得到的图像
  //输出颜色=（3，h,w 填充0.0 float类型）
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  // (P,), 存储每个3D gaussian的半径，多少个点就有多少半径
  //torch::full 是在 C++ 版本的 PyTorch 库中定义的一个函数，
  //创建新张量时保持与 means3D 相同的设置.设备（可能是 GPU 或 CPU），同时将新张量的数据类型设置为 int32。
  torch::Device device(torch::kCUDA);
  //该函数创建了一个torch::Device对象，指定了设备类型为CUDA（即GPU）。
  torch::TensorOptions options(torch::kByte);
  //创建一个torch::TensorOptions对象，其中数据类型为torch::kByte。torch::TensorOptions是
  //PyTorch中用于定义张量属性（如数据类型、内存布局等）的类。
  //在这个例子中，创建的张量选项指定了张量的数据类型为torch::kByte，即8位无符号整型。
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  // (0,), 存储所有3D gaussian对应的参数(均值、尺度、旋转参数、不透明度)的tensor, 会动态分配存储空间
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  //（0,)
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));                
  // (0,), 存储在指定视角下渲染得到的图像的tensor, 会动态分配存储空间
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);               
  // 动态调整 geomBuffer 大小的函数, 并返回对应的数据指针
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);         
  // 动态调整 binningBuffer 大小的函数, 并返回对应的数据指针
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);                 
  // 动态调整 imgBuffer 大小的函数, 并返回对应的数据指针
  // 动态调整 imgBuffer 大小的函数, 并返回对应的数据指针

  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);  //  SH 的维度，16（1+3+5+7）
      }

	        rendered = CudaRasterizer::Rasterizer::forward(
        geomFunc,
        binningFunc,
        imgFunc,
        P, degree, M,                                   // 3D gaussian的个数, 球谐函数的次数, 球谐系数的个数 (球谐系数用于表示颜色)
        background.contiguous().data<float>(),          // 背景颜色, [0, 0, 0]
        W, H,                                           // 图像的宽和高
        means3D.contiguous().data<float>(),             // 每个3D gaussian的XYZ均值
        sh.contiguous().data_ptr<float>(),              // 每个3D gaussian的球谐系数, 用于表示颜色
        colors.contiguous().data<float>(),              // 提前计算好的每个3D gaussian的颜色, []
        opacity.contiguous().data<float>(),             // 每个3D gaussian的不透明度
        scales.contiguous().data_ptr<float>(),          // 每个3D gaussian的XYZ尺度
        scale_modifier,                                 // 尺度缩放系数, 1.0
        rotations.contiguous().data_ptr<float>(),       // 每个3D gaussian的旋转四元组
        cov3D_precomp.contiguous().data<float>(),       // 提前计算好的每个3D gaussian的协方差矩阵, []
        viewmatrix.contiguous().data<float>(),          // 相机外参矩阵, world to camera
        projmatrix.contiguous().data<float>(),          // 投影矩阵, world to image
        campos.contiguous().data<float>(),              // 所有相机的中心点XYZ坐标
        tan_fovx,                                       // 水平视场角一半的正切值
        tan_fovy,                                       // 垂直视场角一半的正切值
        prefiltered,                                    // 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian, False
        out_color.contiguous().data<float>(),           // 在指定的视角下, 对所有3D gaussian进行投影和渲染得到的图像
        radii.contiguous().data<int>(),                 // 存储每个2D gaussian在图像上的半径
        debug);                                         // False
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}


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
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}