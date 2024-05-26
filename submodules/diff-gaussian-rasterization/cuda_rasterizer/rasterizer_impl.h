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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}
//几个fromChunk
//这些函数的作用是从以char数组形式存储的二进制块中读取GeometryState、ImageState、BinningState等类的信息。
	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};
//fromChunk是一个静态成员函数，用于从给定的内存块中创建一个GeometryState对象。它接受一个指向内存块的指针和参数P，并返回一个GeometryState对象。
//该函数在内部解析内存块中的数据，并将其成员变量初始化为相应的值
	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};


	struct BinningState
	{
		size_t sorting_size;                   // 存储用于排序操作的缓冲区大小
		uint64_t* point_list_keys_unsorted;    // 未排序的键列表
		uint64_t* point_list_keys;             // 排序后的键列表
		uint32_t* point_list_unsorted;         // 未排序的点列表
		uint32_t* point_list;                  // 排序后的点列表
		char* list_sorting_space;              // 用于排序操作的缓冲区

		static BinningState fromChunk(char*& chunk, size_t P);
	};
	
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};



