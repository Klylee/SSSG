import os
import sys
import argparse
import numpy as np
import torch
import open3d as o3d
import cv2
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.dataset_readers import readCamerasFromTransforms
from utils.camera_utils import cameraList_from_camInfos
from gaussian_renderer import render
from scene import Scene
from gaussian_renderer import GaussianModel
from scripts.dtu_eval import binary_dilation

def getPointsFromDepth(camera, depth, normal=None, mask=None, scale=1):
    st = int(max(int(scale / 2) - 1, 0))
    depth_view = depth.squeeze()[st::scale, st::scale]
    rays_d = camera.get_rays(scale=scale)
    depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
    pts = rays_d * depth_view[..., None]
    pts = pts[mask]
    R = torch.tensor(camera.R).float().cuda()
    T = torch.tensor(camera.T).float().cuda()
    pts = (pts - T) @ R.transpose(-1,-2)
    if normal != None:
        normal = normal[mask]
        normal = normal @ R.transpose(-1,-2)
    return pts, normal

def normalize_point_cloud(points: np.ndarray):
    if len(points) == 0:
        return pcd
    
    # 计算点云的min和max
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # 计算中心点
    center = (min_coords + max_coords) / 2.0
    
    # 计算当前点云的范围
    current_range = max_coords - min_coords
    max_range = np.max(current_range)
    
    # 避免除以零
    if max_range == 0:
        return pcd
    
    # 平移点云到原点，然后缩放到[-0.5, 0.5]
    normalized_points = (points - center) / max_range
    
    return normalized_points, center, max_range

if __name__ == '__main__':
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--it", default=-1, type=int)
    parser.add_argument('--s', type=str, default='chinesedragon')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    dataset = model.extract(args)
    pipeline = pipeline.extract(args)

    dataset_file = f'/home/yuanyouwen/expdata/neuralto/{args.s}'

    # args.resolution = 1
    # args.ncc_scale = 1

    os.makedirs(f'debug/{args.s}', exist_ok=True)
    all_points = np.empty([0, 3])
    all_normal = np.empty([0, 3])
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(model.extract(args), gaussians, load_iteration=args.it, shuffle=False)
        all_cameras = scene.getTrainCameras()
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # camera = all_cameras[0]
        for camera in tqdm(all_cameras, desc="Rendering"):
            out = render(camera, gaussians, pipeline, background)

            depth = out["plane_depth"].squeeze()
            alpha = out["alpha"].squeeze()
            normal = out["rendered_normal"].permute(1,2,0)
            depth = depth.clone() * camera.mask.float()
            # mask_bg = (alpha > 0.5).float()
            depth = depth #* mask_bg
            depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            depth_i = (depth_i * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(f'debug/{args.s}', camera.image_name + ".jpg"), depth_color)

            points, normal = getPointsFromDepth(camera, depth, normal, camera.mask)
            all_points = np.concatenate((all_points, points.cpu().numpy()), axis=0)
            all_normal = np.concatenate((all_normal, normal.cpu().numpy()), axis=0)

        vertices = torch.from_numpy(all_points).cuda()
        vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1).float()

        sampled_masks = []
            
        visibility = np.ones(vertices.shape[0], dtype=bool)

        for i, camera in tqdm(enumerate(all_cameras),  desc="Culling mesh given masks"):      
            with torch.no_grad():
                # transform and project
                cam_points =  vertices @ camera.full_proj_transform
                ndc_points = cam_points[:, :2] / (cam_points[:, 2].unsqueeze(1) + 1e-6)
                point2d_x = (ndc_points[:, 0] + 1) * 0.5 * camera.image_width
                point2d_y = (ndc_points[:, 1] + 1) * 0.5 * camera.image_height
                point2d = torch.stack([point2d_x, point2d_y], dim=-1)

                in_image = (point2d[:, 0] >= 0) & (point2d[:, 0] < camera.image_width) & (point2d[:, 1] >= 0) & (point2d[:, 1] < camera.image_height)
                point2d[:, 0] = point2d[:, 0].clamp(0, camera.image_width - 1)
                point2d[:, 1] = point2d[:, 1].clamp(0, camera.image_height - 1)
                
                mask = binary_dilation((camera.mask).float(), 1).bool()
                # mask = camera.mask
                valid = in_image & mask[point2d[:, 1].long(), point2d[:, 0].long()]
                visibility &= valid.cpu().numpy()

        all_points = all_points[visibility]
        all_normal = all_normal[visibility]
    
    # all_points, center, max_range = normalize_point_cloud(all_points)
    # print(f'trans: {center}, {max_range}')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.normals = o3d.utility.Vector3dVector(all_normal)
    pcd = pcd.voxel_down_sample(voxel_size=0.002)
    pcd.orient_normals_consistent_tangent_plane(k=30)
    print(len(pcd.points))
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #         pcd, depth=9)
    # o3d.io.write_triangle_mesh(f"debug/{args.s}_mesh.obj", mesh)
    o3d.io.write_point_cloud(f'debug/{args.s}_recon.ply', pcd)