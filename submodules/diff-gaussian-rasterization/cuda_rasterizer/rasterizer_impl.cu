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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
//寻找给定无符号整数 n 的最高有效位（Most Significant Bit, MSB）的下一个最高位
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
// 这个方法是一个包装器（Wrapper），其目的是调用辅助的   粗略视锥体包含测试
//（coarse frustum containment test）
__global__ void checkFrustum(int P, // 需要检查的点的个数
	const float* orig_points, // 需要检查的点的世界坐标
	const float* viewmatrix, // W2C矩阵
	const float* projmatrix, // 投影矩阵
	bool* present) // 返回值，表示能不能被看见
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
//为所有高斯/瓦片重叠生成一个键/值对。每高斯运行一次（1:N映射）
__global__ void duplicateWithKeys(
	int P, //3D gaussian的个数
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();//获取当前线程在该网格内的唯一标识
	if (idx >= P) // 线程索引，该显线程处理第idx个Gaussian
		return;

	// Generate no key/value pair for invisible Gaussians
	//为可见的高斯生成键/值对
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		//在用于写入键/值的缓冲区中查找此高斯偏移量。
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;
		//getRect计算当前的2D gaussian落在哪几个tile上
		//因为要给Gaussian覆盖的每个tile生成一个(key, value)对，
		// 所以先获取它占了哪些tile
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		// // //对于边界矩形重叠的每个图块，发射一个键/值对。
		// // 键是|tile ID|depth|，值是高斯的ID。
		// // 使用此键对值进行排序会产生列表中的高斯ID，
		// // 因此它们首先按瓦片排序，然后按深度排序。
		for (int y = rect_min.y; y < rect_max.y; y++)
		//这个循环用于遍历从rect_min.y到rect_max.y - 1（不包括rect_max.y）的所有整数。
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x; // tile的ID
				key <<= 32;// 放在高位
				key |= *((uint32_t*)&depths[idx]);// 低位是深度
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;// 数组中的偏移量
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
//检查关键点以查看它是否位于完全排序列表中某个tile范围的开始/结束处。
//如果是，请写入此tile的开始/结束。每个实例化（重复）高斯ID运行一次。
// 识别每个瓦片（tile）在排序后的高斯ID列表中的范围
// 目的是确定哪些高斯ID属于哪个瓦片，并记录每个瓦片的开始和结束位置

__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;//大于线程就停止运行

	// Read tile ID from key. Update start/end of tile range if at limit.
	//从钥读取tileID。如果达到限制，则更新tile范围的开始/结束。
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;//将其键值（key）右移32位得到当前分组的tile（currtile）
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}
// 它从索引idx - 1处获取上一个点的编号prevtile，
// 然后判断当前点currtile是否与上一个点属于不同的
// 组。如果是，则更新上一个组的结束索引为idx，
// 并更新当前组的起始索引为idx。最后，如果idx等于列表长度L - 1，
// 则更新当前组的结束索引为L。这样可以记录每个组的起始和结束索引，
// 以便后续处理。

// Mark Gaussians as visible/invisible, based on view frustum testing
//根据视锥体测试（view frustum testing）来标记高斯分布（可能是
//3D空间中的点云或其他数据结构）为可见或不可见。
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}
//该函数的功能是从chunk中提取数据，并将其存储在创建的GeometryState对象geom中。
//通过调用obtain函数多次从chunk中提取不同数据，并将其存储在geom的不同成员变量中，
//包括depths、clamped、internal_radii、means2D、cov3D、conic_opacity、rgb和tiles_touched。这些成员变量都是GeometryState类的公有成员。

//然后，使用cub::DeviceScan的InclusiveSum函数计算geom.tiles_touched
//数组的累加和，并将结果存储在同一数组中。


//函数是一个类的静态成员函数，用于从一个字符指针中获取图像状态信息，
//并返回一个ImageState对象。函数通过调用obtain函数三次，
//分别从chunk中读取accum_alpha、n_contrib和ranges三个成员变量的值，
//并将其赋值给新创建的ImageState对象img的对应成员变量。最后返回img对象。
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}


// 初始化 BinningState 实例，分配所需的内存，并执行排序操作
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.point_list_keys, P, 128);
    obtain(chunk, binning.point_list_keys_unsorted, P, 128);
    // 在 GPU 上进行基数排序, 将 point_list_keys_unsorted 作为键，point_list_unsorted 作为值进行排序，
	//排序结果存储在 point_list_keys 和 point_list 中
    cub::DeviceRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.point_list_keys_unsorted, binning.point_list_keys,
        binning.point_list_unsorted, binning.point_list, P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}
//该代码片段使用了CUB库中的DeviceRadixSort类的
// SortPairs函数进行对点列表的键值对排序。
// 其中，nullptr表示不使用用户定义的比较函数；
// binning.sorting_size表示要排序的数据数量；
// binning.point_list_keys_unsorted和binning.point_list_keys
// 分别表示未排序和排序后的键数组；binning.point_list_unsorted和binning.point_list
// 分别表示未排序和排序后的值数组；P是点数
// //接着，通过obtain函数获取排序后的列表的存储空间。最后，返回排序后的binning对象。

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(  // 可以把这个当成 main 函数
//在CudaRasterizer命名空间内的Rasterizer类的一个名为forward的成员函数
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	bool debug)
{
    const float focal_y = height / (2.0f * tan_fovy);   // 垂直方向的焦距 focal_y
    const float focal_x = width / (2.0f * tan_fovx);    // 水平方向的焦距 focal_x

    size_t chunk_size = required<GeometryState>(P);     
	// 计算存储所有3D gaussian的各个参数所需要的空间大小
    char* chunkptr = geometryBuffer(chunk_size);        
	// 给所有3D gaussian的各个参数分配存储空间, 并返回存储空间的指针
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);  
	// 在给定的内存块中初始化 GeometryState 结构体,
    //为不同成员分配空间，并返回一个初始化的实例
	//用于存储2D高斯的信息

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;// 指向radii数据的指针
	}
    // 定义了一个三维网格（dim3 是 CUDA 中定义三维网格维度的数据类型），确定了在水平和垂直方向上需要多少个块来覆盖整个渲染区域
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// 确定了每个块在 X（水平）和 Y（垂直）方向上的线程数
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	//在训练过程中动态调整基于图像的辅助缓冲区的大小。
    size_t img_chunk_size = required<ImageState>(width * height);               // 计算存储所有2D pixel的各个参数所需要的空间大小
    char* img_chunkptr = imageBuffer(img_chunk_size);                           // 给所有2D pixel的各个参数分配存储空间, 并返回存储空间的指针
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);  // 在给定的内存块中初始化 ImageState 结构体, 为不同成员分配空间，并返回一个初始化的实例


	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}
//执行针对每个高斯分布的预处理步骤，包括：转换操作、边界限制以及从球谐函数（SHs）到RGB颜色空间的转换。
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    CHECK_CUDA(FORWARD::preprocess(
        P, D, M,                      // 3D gaussian的个数, 球谐函数的次数, 球谐系数的个数 (球谐系数用于表示颜色)
        means3D,                      // 每个3D gaussian的XYZ均值
        (glm::vec3*)scales,           // 每个3D gaussian的XYZ尺度
        scale_modifier,               // 尺度缩放系数, 1.0
        (glm::vec4*)rotations,        // 每个3D gaussian的旋转四元组
        opacities,                    // 每个3D gaussian的不透明度
        shs,                          // 每个3D gaussian的球谐系数, 用于表示颜色
        geomState.clamped,            // 存储每个3D gaussian的R、G、B是否小于0
        cov3D_precomp,                // 提前计算好的每个3D gaussian的协方差矩阵, []
        colors_precomp,               // 提前计算好的每个3D gaussian的颜色, []
        viewmatrix,                   // 相机外参矩阵, world to camera
        projmatrix,                   // 投影矩阵, world to image
        (glm::vec3*)cam_pos,          // 所有相机的中心点XYZ坐标
        width, height,                // 图像的宽和高
        focal_x, focal_y,             // 水平、垂直方向的焦距
        tan_fovx, tan_fovy,           // 水平、垂直视场角一半的正切值
        radii,                        // 存储每个2D gaussian在图像上的半径
        geomState.means2D,            // 存储每个2D gaussian的均值
        geomState.depths,             // 存储每个2D gaussian的深度
        geomState.cov3D,              // 存储每个3D gaussian的协方差矩阵
        geomState.rgb,                // 存储每个2D pixel的颜色
        geomState.conic_opacity,      // 存储每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
        tile_grid,                    // 在水平和垂直方向上需要多少个块来覆盖整个渲染区域
        geomState.tiles_touched,      // 存储每个2D gaussian覆盖了多少个tile
        prefiltered                   // 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian
    ), debug)


	// ---开始--- 通过视图变换 W 计算出像素与所有重叠高斯的距离，即这些高斯的深度，形成一个有序的高斯列表
	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
//该函数使用cub库的DeviceScan类的InclusiveSum方法，在GPU上进行就地累加和扫描操作。将扫描空间、扫描大小、瓦片触摸计数和点偏移量作为输入，将累加和存储在点偏移量数组中。
//在调用之前和之后，分别进行CUDA错误检查，并在调试模式下输出错误信息。
	
	
	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	//获取要启动的高斯实例总数，并根据这个数量调整辅助缓冲区的大小。
	int num_rendered;  // 存储所有的2D gaussian总共覆盖了多少个tile
    // 将 geomState.point_offsets 数组中最后一个元素的值复制到主机内存中的变量 num_rendered
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);



	size_t binning_chunk_size = required<BinningState>(num_rendered);
	//根据渲染的元素数量num_rendered，计算并返回一个用于分块聚类的大小。
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	//内存中申请一块大小为binning_chunk_size的缓冲区，用于存储图像的binning处理结果。
	//返回一个指向该缓冲区起始位置的字符指针。
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	//对于每个需要渲染的实例，函数将生成合适的[瓷砖 | 深度]键（tile|depth key）
	//以及与之对应的一组重复的高斯索引。这些索引会被排序，
	// 将每个3D gaussian的对应的tile index和深度存到point_list_keys_unsorted中
    // 将每个3D gaussian的对应的index（第几个3D gaussian）存到point_list_unsorted中

	duplicateWithKeys << <(P + 255) / 256, 256 >> > (  // 根据 tile，复制 Gaussian
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	// 按关键字对（重复的）高斯索引的完整列表进行排序
	// 对一个键值对列表进行排序。这里的键值对由 binningState.point_list_keys_unsorted 和 binningState.point_list_unsorted 组成
    // 排序后的结果存储在 binningState.point_list_keys 和 binningState.point_list 中
    // binningState.list_sorting_space 和 binningState.sorting_size 指定了排序操作所需的临时存储空间和其大小
    // num_rendered 是要排序的元素总数。0, 32 + bit 指定了排序的最低位和最高位，这里用于确保排序考虑到了足够的位数，以便正确处理所有的键值对

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs( // 对复制后的所有 Gaussians 进行排序，排序的结果可供平行化渲染使用
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)
    // 将 imgState.ranges 数组中的所有元素设置为 0
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	// 在排序列表中确定每个磁贴工作负载的开始和结束
	// 识别每个瓦片（tile）在排序后的高斯ID列表中的范围
    // 目的是确定哪些高斯ID属于哪个瓦片，并记录每个瓦片的开始和结束位置
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (  
			// 根据有序的Gaussian列表，判断每个 tile 需要跟哪一个 
			//range 内的 Gaussians 进行计算
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)
	// ---结束--- 通过视图变换 W 计算出像素与所有重叠高斯的距离，即这些高斯的深度，形成一个有序的高斯列表

	// Let each tile blend its range of Gaussians independently in parallel
	//让每个瓦片并行地独立地混合其高斯级数范围
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(  // 核心渲染函数，具体实现在 forward.cu/renderCUDA
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}