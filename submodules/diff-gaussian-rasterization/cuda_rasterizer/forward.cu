/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 结构：
 * 定义预处理
 * 定义渲染
 * 调用预处理
 * 调用渲染
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH
(
	int idx, // 该线程负责第几个Gaussian
	int deg, // 球谐的度数
	int max_coeffs, // 一个Gaussian最多有几个傅里叶系数
	const glm::vec3* means, // Gaussian中心位置
	glm::vec3 campos, // 相机位置
	const float* shs, // 球谐系数
	bool* clamped) // 表示每个值是否被截断了（RGB只能为正数），这个在反向传播的时候用
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);//即观察方向

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	////RGB颜色被钳制为正值。如果值为
//被夹住了，我们需要在后传球时注意这一点。
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D


	(const float3& mean, // Gaussian中心坐标
	float focal_x, // x方向焦距
	float focal_y, // y方向焦距
	float tan_fovx,
	float tan_fovy,
	const float* cov3D, // 已经算出来的三维协方差矩阵
	const float* viewmatrix) // W2C矩阵
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	//以下为“EWA飞溅”（Zwicker等人，2002）中方程29和31概述的步骤建模。
	//此外还考虑了视口的纵横比/缩放。用于说明行/列主要约定的转换。
	float3 t = transformPoint4x3(mean, viewmatrix);// W2C矩阵乘Gaussian中心坐标得其在相机坐标系下的坐标

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;// Gaussian中心在像平面上的x坐标
	t.y = min(limy, max(-limy, tytz)) * t.z;// Gaussian中心在像平面上的y坐标

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);// 雅可比矩阵（用泰勒展开近似）

	glm::mat3 W = glm::mat3(// W2C矩阵
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3( // 3D协方差矩阵，是对称阵
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
		// transpose(J) @ transpose(W) @ Vrk @ W @ J

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	// 协方差矩阵是对称的，只用存储上三角，故只返回三个数
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(
	const glm::vec3 scale, // 表示缩放的三维向量
	float mod, // 对应gaussian_renderer/__init__.py中的scaling_modifier
	const glm::vec4 rot, // 表示旋转的四元数
	float* cov3D) // 结果：三维协方差矩阵
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;//论文里的R.T S.T S R

	// Covariance is symmetric, only store upper right
	////协方差是对称的，只存储右上角
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,  // Gaussian中心在图像上的像素坐标
	float* depths, // Gaussian中心的深度，即其在相机坐标系的z轴的坐标
	float* cov3Ds, // 三维协方差矩阵
	float* rgb, // 根据球谐算出的RGB颜色值
	float4* conic_opacity, // 椭圆对应二次型的矩阵和不透明度的打包存储
	const dim3 grid, // tile的在x、y方向上的数量
	uint32_t* tiles_touched, // Gaussian覆盖的tile数量
	bool prefiltered)

{
	// 每个线程处理一个3D gaussian, index超过3D gaussian总数的线程直接返回, 防止数组越界访问
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	////将半径和碰到的tile初始化为0。
	//如果这一点没有改变，这个高斯将不会被进一步处理。
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside. 给定指定的相机姿势，此步骤确定哪些3D高斯位于相机的视锥体之外。
	//这样做可以确保在后续计算中不涉及给定视图之外的3D高斯，从而节省计算资源。
	// 判断当前处理的3D gaussian的中心点(均值XYZ)是否在视锥（frustum）内, 如果不在则直接返回
	float3 p_view;// 用于存储将 p_orig 通过视图矩阵 viewmatrix 
	//将当前3D gaussian的中心点投影到相机坐标系(转换到视图空间后的点坐标)
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting  以下代码将3D高斯（椭球）被投影到2D图像空间（椭圆），存储必要的变量供后续渲染使用
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
		// 将当前3D gaussian的中心点从世界坐标系投影到裁剪坐标系
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// 想要除以p_hom.w从而转成正常的3D坐标，这里防止除零
	// 将当前3D gaussian的中心点从裁剪坐标转变到归一化设备坐标（Normalized Device Coordinates, NDC）
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. //如果三维协方差矩阵是预先计算的，/
	//请使用它，否则根据缩放和旋转参数进行计算。
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// 将当前的3D gaussian投影到2D图像，得到对应的2D gaussian的协方差矩阵cov
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	    // 计算当前2D gaussian的协方差矩阵cov的逆矩阵conic
		//det 是行列式
	float det = (cov.x * cov.z - cov.y * cov.y); // 二维协方差矩阵的行列式
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;// 行列式的逆
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
//函数计算了给定协方差矩阵的逆矩阵，并将其存储在conic变量中。
//如果协方差矩阵的行列式为零，则函数直接返回。
//行列式的逆用于计算逆矩阵的元素。最后，将逆矩阵的元素存储在conic中，
//以便后续使用。
	// conic是cone的形容词，意为“圆锥的”。猜测这里是指圆锥曲线（椭圆）。
		// 二阶矩阵求逆口诀：“主对调，副相反”。




	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	//计算屏幕空间中的范围（通过找到2D协方差矩阵的特征值）。
	//使用范围来计算此高斯重叠的屏幕空间平铺的边界矩形。
	//如果矩形覆盖0个tile，则退出。
	    // 计算2D gaussian的协方差矩阵cov的特征值lambda1, lambda2, 
		//从而计算2D gaussian的最大半径
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// 韦达定理求二维协方差矩阵的特征值
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
		// 这里就是截取Gaussian的中心部位（3σ原则），只取像平面上半径为my_radius的部分
//.f 是一个后缀，用于明确指定数值3是一个浮点数
	// 对协方差矩阵进行特征值分解时，
	//可以得到描述分布形状的主轴（特征向量）以及这些轴上分布的宽度（特征值）
//

	    // 将归一化设备坐标（Normalized Device Coordinates, NDC）转换为像素坐标
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	//getrect函数计算当前的2D gaussian落在哪几个tile上
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	    // 如果没有命中任何一个title则直接返回
		// 检查该Gaussian在图片上覆盖了哪些tile（由一个tile组成的矩形表示）
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;



	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
//如果颜色已经预先计算，请使用它们，否则将球面谐波系数转换为RGB颜色。
	if (colors_precomp == nullptr)
	// 从每个3D gaussian对应的球谐系数中计算对应的颜色
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	//为接下来的步骤存储一些有用的帮助程序数据。
	depths[idx] = p_view.z;// 深度，即相机坐标系的z轴
	radii[idx] = my_radius;// Gaussian在像平面坐标系下的半径
	points_xy_image[idx] = point_image;// Gaussian中心在图像上的像素坐标
	// Inverse 2D covariance and opacity neatly pack into one float4
	//逆2D协方差和不透明度巧妙地组合成一个float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };  
	// 前三个将被用于计算高斯的指数部分从而得到 prob（查询点到该高斯的距离->prob，
	//例如，若查询点位于该高斯的中心则 prob 为 1）。最后一个是该高斯本身的密度。
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}	// Gaussian覆盖的tile数量

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 线程在读取数据（把数据从公用显存拉到block自己的显存）和进行计算之间来回切换，
// 使得线程们可以共同读取Gaussian数据。
// 这样做的原因是block共享内存比公共显存快得多。
template <uint32_t CHANNELS>// CHANNELS取3，即RGB三个通道
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)// 这是 CUDA 启动核函数时使用的线程格和线程块的数量。
renderCUDA(
	const uint2* __restrict__ ranges,// 每个tile对应排过序的数组中的哪一部分
	const uint32_t* __restrict__ point_list,// 按tile、深度排序后的Gaussian ID列表
	int W, int H,// 图像宽高
	const float2* __restrict__ points_xy_image,// 图像上每个Gaussian中心的2D坐标
	const float* __restrict__ features,// RGB颜色
	const float4* __restrict__ conic_opacity,//二维协方差矩阵和透明度
	float* __restrict__ final_T,// 最终的透光率
	uint32_t* __restrict__ n_contrib,
	// 多少个Gaussian对该像素的颜色有贡献（用于反向传播时判断各个Gaussian有没有梯度）
	const float* __restrict__ bg_color,//背景颜色
	float* __restrict__ out_color)//// 渲染结果（图片）
{
	// Identify current tile and associated min/max pixel range.
	// 1.确定当前像素范围：
	// 这部分代码用于确定当前线程块要处理的像素范围，包括 pix_min 和 pix_max，并计算当前线程对应的像素坐标 pix。
	//确定当前瓦片及其关联的最小/最大像素范围。
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X; // x方向上tile的个数
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
		// 负责的tile的坐标较小的那个角的坐标 // 当前处理的tile的左上角的像素坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
		// 负责的tile的坐标较大的那个角的坐标 // 当前处理的tile的右下角的像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
		// 负责某个像素的坐标
	uint32_t pix_id = W * pix.y + pix.x;
		// 负责某个像素的id
	float2 pixf = { (float)pix.x, (float)pix.y }; // pix的浮点数版本



	// Check if this thread is associated with a valid pixel or outside.
	//检查此线程是否与有效的像素或外部相关联。像素有没有跑到图像外面去
	//// 2.判断当前线程是否在有效像素范围内：
	// 根据像素坐标判断当前线程是否在有效的图像范围内，如果不在，则将 done 设置为 true，表示该线程无需执行渲染操作。
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	///完成的线程可以帮助获取，但不要光栅化
	bool done = !inside;


	// Load start/end range of IDs to process in bit sorted list.
	//加载要处理的ID在位排序列表中的起始/结束范围
	// 3.加载点云数据处理范围：
	// 这部分代码加载当前线程块要处理的点云数据的范围，即 range 数组中对应的范围，并计算点云数据的迭代批次 rounds 和总共要处理的点数 toDo。
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
		// BLOCK_SIZE = 16 * 16 = 256
		// 我要把任务分成rounds批，每批处理BLOCK_SIZE个Gaussians
		// 每一批，每个线程负责读取一个Gaussian的信息，
		// 所以该block的256个线程每一批就可以读取256个Gaussian的信息
	int toDo = range.y - range.x;// 我要处理的Gaussian个数

	// Allocate storage for batches of collectively fetched data.  
	//每个线程取一个，并行读数据到 shared memory。然后每个线程都访问该shared memory，读取顺序一致。
	// __shared__: 同一block中的线程共享的内存
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	// 5.初始化渲染相关变量：
	// 初始化渲染所需的一些变量，包括、贡献者数量等
	float T = 1.0f;//透光率
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };




	// 6.迭代处理点云数据：
	// 在每个迭代中，处理一批点云数据。内部循环迭代每个点，进行基于锥体参数的渲染计算，并更新颜色信息。
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	//代码使用 rounds 控制循环的迭代次数，每次迭代处理一批点云数据//
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		// 检查是否所有线程块都已经完成渲染：
		// 通过 __syncthreads_count 统计已经完成渲染的线程数，如果整个线程块都已完成，则跳出循环。
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		//集体地从全局内存中将每个高斯分布的数据读取到共享内存中。
		----
		//共享内存中获取点云数据：
		//每个线程通过索引 progress 计算要加载的点云数据的索引 coll_id，
		//然后从全局内存中加载到共享内存 collected_id、collected_xy 和 collected_conic_opacity 中。
		//block.sync() 确保所有线程都加载完成。

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch// 迭代处理当前批次的点云数据
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range跟踪在指定范围内的当前位置。
			contributor++;

			// Resample using conic matrix (cf. "Surface //使用XX矩阵进行重采样
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];                     // 当前处理的2D gaussian在图像上的中心点坐标
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };     // 当前处理的2D gaussian的中心点到当前处理的pixel的offset
            float4 con_o = collected_conic_opacity[j];       // 当前处理的2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
			 // 计算高斯分布的强度（或权重），用于确定像素在光栅化过程中的贡献程度
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// 来自三维高斯溅射论文的方程(2)。通过将高斯不透明度与其从均值开始的指数衰减相乘来获取alpha值。
			// 注意要避免数值不稳定问题（参见论文附录）。
			float alpha = min(0.99f, con_o.w * exp(power));  // opacity * 像素点出现在这个高斯的几率
			//	// Gaussian对于这个像素点来说的不透明度
			// 注意con_o.w是”opacity“，是Gaussian整体的不透明度
			if (alpha < 1.0f / 255.0f)  // 太小了就当成透明的
				continue;
			float test_T = T * (1 - alpha);  // alpha合成的系数
			if (test_T < 0.0001f)  // 累乘不透明度到一定的值，标记这个像素的渲染结束
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			// 使用高斯分布进行渲染计算：更新颜色信息 C。
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;// 更新透光率

			// Keep track of last range entry to update this pixel.
			//跟踪更新该像素的最后一个范围条目。
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	//7. 写入最终渲染结果：
	// 如果当前线程在有效像素范围内，则将最终的渲染结果写入相应的缓冲区，包括 final_T、n_contrib 和 out_color
	if (inside)
	{
		final_T[pix_id] = T;  // 用于反向传播计算梯度/ 渲染过程后每个像素的最终透明度或透射率值
		n_contrib[pix_id] = last_contributor;  // 记录数量，用于提前停止计算
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}
//渲染过程
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}// 一个线程负责一个像素，一个block负责一个tile


void FORWARD::preprocess(int P, int D, int M,
    const float* means3D,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    int* radii,
    float2* means2D,
    float* depths,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered)
{
    preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
        P, D, M,               // 3D gaussian的个数, 球谐函数的次数, 球谐系数的个数 (球谐系数用于表示颜色)
        means3D,               // 每个3D gaussian的XYZ均值
        scales,                // 每个3D gaussian的XYZ尺度
        scale_modifier,        // 尺度缩放系数, 1.0
        rotations,             // 每个3D gaussian的旋转四元组
        opacities,             // 每个3D gaussian的不透明度
        shs,                   // 每个3D gaussian的球谐系数, 用于表示颜色
        clamped,               // 存储每个3D gaussian的R、G、B是否小于0
        cov3D_precomp,         // 提前计算好的每个3D gaussian的协方差矩阵, []
        colors_precomp,        // 提前计算好的每个3D gaussian的颜色, []
        viewmatrix,            // 相机外参矩阵, world to camera
        projmatrix,            // 投影矩阵, world to image
        cam_pos,               // 所有相机的中心点XYZ坐标
        W, H,                  // 图像的宽和高
        tan_fovx, tan_fovy,    // 水平、垂直视场角一半的正切值
        focal_x, focal_y,      // 水平、垂直方向的焦距
        radii,                 // 存储每个2D gaussian在图像上的半径
        means2D,               // 存储每个2D gaussian的均值
        depths,                // 存储每个2D gaussian的深度
        cov3Ds,                // 存储每个3D gaussian的协方差矩阵
        rgb,                   // 存储每个2D pixel的颜色
        conic_opacity,         // 存储每个2D gaussian的协方差矩阵的逆矩阵以及它的不透明度
        grid,                  // 在水平和垂直方向上需要多少个tile来覆盖整个渲染区域
        tiles_touched,         // 存储每个2D gaussian覆盖了多少个tile
        prefiltered            // 是否预先过滤掉了中心点(均值XYZ)不在视锥（frustum）内的3D gaussian
        );
}