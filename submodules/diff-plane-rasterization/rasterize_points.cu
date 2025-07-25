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
#include "cuda_rasterizer/utils.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t)
{
	auto lambda = [&t](size_t N)
	{
		t.resize_({(long long)N});
		return reinterpret_cast<char *>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor &background,
	const torch::Tensor &means3D,
	const torch::Tensor &colors,
	const torch::Tensor &opacity,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &all_map,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const bool prefiltered,
	const bool render_geo,
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor out_observe = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
	torch::Tensor out_all_map = torch::full({NUM_ALL_MAP, H, W}, 0, float_opts);
	torch::Tensor out_plane_depth = torch::full({1, H, W}, 0, float_opts);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (P != 0)
	{
		int M = 0;
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}

		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			background.contiguous().data_ptr<float>(),
			W, H,
			means3D.contiguous().data_ptr<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data_ptr<float>(),
			opacity.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data_ptr<float>(),
			all_map.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_color.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			out_observe.contiguous().data_ptr<int>(),
			out_all_map.contiguous().data_ptr<float>(),
			out_plane_depth.contiguous().data_ptr<float>(),
			render_geo,
			debug);
	}
	return std::make_tuple(rendered, out_color, radii, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor &background,
	const torch::Tensor &all_map_pixels,
	const torch::Tensor &plane_depth_pixels,
	const torch::Tensor &means3D,
	const torch::Tensor &radii,
	const torch::Tensor &colors,
	const torch::Tensor &all_maps,
	const torch::Tensor &scales,
	const torch::Tensor &rotations,
	const float scale_modifier,
	const torch::Tensor &cov3D_precomp,
	const torch::Tensor &viewmatrix,
	const torch::Tensor &projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor &dL_dout_color,
	const torch::Tensor &dL_dout_all_map,
	const torch::Tensor &dL_dout_plane_depth,
	const torch::Tensor &sh,
	const int degree,
	const torch::Tensor &campos,
	const torch::Tensor &geomBuffer,
	const int R,
	const torch::Tensor &binningBuffer,
	const torch::Tensor &imageBuffer,
	const bool render_geo,
	const bool debug)
{
	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	int M = 0;
	if (sh.size(0) != 0)
	{
		M = sh.size(1);
	}

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D_abs = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans3D_view = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_dall_map = torch::zeros({P, NUM_ALL_MAP}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::backward(
			P, degree, M, R,
			background.contiguous().data_ptr<float>(),
			all_map_pixels.contiguous().data_ptr<float>(),
			plane_depth_pixels.contiguous().data_ptr<float>(),
			W, H,
			means3D.contiguous().data_ptr<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data_ptr<float>(),
			all_maps.contiguous().data_ptr<float>(),
			scales.data_ptr<float>(),
			scale_modifier,
			rotations.data_ptr<float>(),
			cov3D_precomp.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			radii.contiguous().data_ptr<int>(),
			reinterpret_cast<char *>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char *>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char *>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data_ptr<float>(),
			dL_dout_all_map.contiguous().data_ptr<float>(),
			dL_dout_plane_depth.contiguous().data_ptr<float>(),
			dL_dmeans2D.contiguous().data_ptr<float>(),
			dL_dmeans2D_abs.contiguous().data_ptr<float>(),
			dL_dmeans3D_view.contiguous().data_ptr<float>(),
			dL_dconic.contiguous().data_ptr<float>(),
			dL_dopacity.contiguous().data_ptr<float>(),
			dL_dcolors.contiguous().data_ptr<float>(),
			dL_dmeans3D.contiguous().data_ptr<float>(),
			dL_dcov3D.contiguous().data_ptr<float>(),
			dL_dsh.contiguous().data_ptr<float>(),
			dL_dscales.contiguous().data_ptr<float>(),
			dL_drotations.contiguous().data_ptr<float>(),
			dL_dall_map.contiguous().data_ptr<float>(),
			render_geo,
			debug);
	}

	return std::make_tuple(dL_dmeans2D, dL_dmeans2D_abs, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dall_map);
}

torch::Tensor markVisible(
	torch::Tensor &means3D,
	torch::Tensor &viewmatrix,
	torch::Tensor &projmatrix)
{
	const int P = means3D.size(0);

	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(
			P,
			means3D.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			present.contiguous().data_ptr<bool>());
	}

	return present;
}

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor &opacity_old,
	torch::Tensor &scale_old,
	torch::Tensor &N,
	torch::Tensor &binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);

	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	if (P != 0)
	{
		UTILS::ComputeRelocation(
			P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);
}