import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse, os, csv

print("Current working directory:", os.getcwd())
import sys
sys.path.append(os.getcwd())
from scene.dataset_readers import readCamerasFromTransforms
from arguments import ModelParams

from pathlib import Path
from utils.camera_utils import cameraList_from_camInfos
import torch
import torch.nn.functional as F
import trimesh

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


def create_disk_kernel(radius, device="cuda"):
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1, device=device),
                          torch.arange(-radius, radius + 1, device=device),
                          indexing="ij")
    dist = torch.sqrt(x**2 + y**2)
    kernel = (dist <= radius).float()
    return kernel

def binary_dilation(mask, radius, iterations=1):
    """
    Perform binary dilation with a disk-shaped kernel.
    
    Args:
        mask (torch.Tensor): Binary mask of shape [H, W] or [B, 1, H, W].
        radius (int): Radius of the disk kernel.
        iterations (int): Number of dilation iterations.
        
    Returns:
        torch.Tensor: Dilated binary mask with the same shape as input.
    """
    # Create the disk kernel
    kernel = create_disk_kernel(radius, device=mask.device).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Ensure mask is 4D: [B, 1, H, W]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)

    # Perform dilation
    for _ in range(iterations):
        mask = F.conv2d(mask.float(), kernel, padding=radius)
        mask = (mask > 0).float()  # Binarize the result

    return mask.squeeze(0).squeeze(0)

def knn_approximation(points1, points2, k=1, batch_size=1024):
    N1 = points1.shape[0]
    distances = []

    for i in range(0, N1, batch_size): # chunk the input to avoid OOM
        chunk = points1[i:i + batch_size]  # [batch_size, 3]
        dists = torch.cdist(chunk, points2)  # [batch_size, N2]
        min_dist, _ = torch.topk(dists, k=k, largest=False)
        distances.append(min_dist)
    
    return torch.cat(distances, dim=0)

def chamfer_distance(points1: torch.Tensor, points2: torch.Tensor) -> float:
    diff_1 = knn_approximation(points1, points2)  # [N1, N2]

    min_dist_1, _ = torch.min(diff_1, dim=1)
    min_dist_2, _ = torch.min(diff_1, dim=0)

    cd = torch.mean(min_dist_1) + torch.mean(min_dist_2)
    return cd.item()

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    parser.add_argument('--data', type=str, default='data_in.ply')
    parser.add_argument('--scan', type=int, default=1)
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--dataset_dir', type=str, default='.')
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=10)
    args = parser.parse_args()

    dataset_file = '/home/yuanyouwen/expdata/neuralto/chinesedragon'
    output_file = 'output_neuralto/chinesedragon/test'
    gt_mesh_file = '/home/yuanyouwen/expdata/neuralto/gt_mesh/dragon.obj'

    reconstructed_mesh_file = 'tsdf_fusion_post.ply'

    if not os.path.exists(gt_mesh_file):
        print('not exists gt mesh file')
    
    # load masks
    train_cam_infos = readCamerasFromTransforms(dataset_file, "transforms_train.json", white_background=False)
    test_cam_infos = readCamerasFromTransforms(dataset_file, "transforms_test.json", white_background=False)
    train_cameras = cameraList_from_camInfos(train_cam_infos, 1.0, lp.extract(args))
    test_cameras = cameraList_from_camInfos(test_cam_infos, 1.0, lp.extract(args))
    all_cameras = train_cameras + test_cameras

    mask_cull = True
    
    
    thresh = args.downsample_density
    if args.mode == 'mesh':
        reconstructed_mesh = trimesh.load_mesh(os.path.join(output_file, 'mesh', reconstructed_mesh_file))

        if mask_cull:
            # project and filter
            vertices = torch.from_numpy(reconstructed_mesh.vertices).cuda()
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
                    
                    mask = binary_dilation((camera.mask).float(), 8).bool()
                    valid = in_image & mask[point2d[:, 1].long(), point2d[:, 0].long()]
                    visibility &= valid.cpu().numpy()

                    # point_valid = point2d[valid]
                    # image = torch.zeros([camera.image_height, camera.image_width], dtype=torch.uint8)
                    # image[point_valid[:, 1].long(), point_valid[:, 0].long()] = 255
                    # cv2.imwrite(f"debug/{camera.image_name}.png", image.cpu().numpy())

            new_vertices = reconstructed_mesh.vertices[visibility]
            old_to_new_map = -np.ones(reconstructed_mesh.vertices.shape[0], dtype=np.int64)
            old_to_new_map[visibility] = np.arange(np.sum(visibility))
            new_faces_mask = visibility[reconstructed_mesh.faces].all(axis=1)
            new_faces = old_to_new_map[reconstructed_mesh.faces[new_faces_mask]]

            assert np.all(new_faces >= 0)
            assert np.all(new_faces < len(new_vertices))


            reconstructed_mesh = o3d.geometry.TriangleMesh()
            reconstructed_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
            reconstructed_mesh.triangles = o3d.utility.Vector3iVector(new_faces)
            o3d.io.write_triangle_mesh(os.path.join(output_file, 'mesh', 'filtered_' + reconstructed_mesh_file.replace('.ply', '.obj')), reconstructed_mesh)
            
        
        # align points cloud
        p_pcd = o3d.geometry.PointCloud()
        p_pcd.points = o3d.utility.Vector3dVector(new_vertices)
        
        if Path(gt_mesh_file).suffix == '.ply':
            gt_pcd = o3d.io.read_point_cloud(gt_mesh_file)
        elif Path(gt_mesh_file).suffix == '.obj':
            gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_file)
            gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=100_000)
        
        reg = o3d.pipelines.registration.registration_icp(
            p_pcd,
            gt_pcd,
            10.0,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-6, 50),
        )
        reg2 = o3d.pipelines.registration.registration_icp(
            p_pcd,
            gt_pcd,
            2.5,
            reg.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-6, 50),
        )
        reg3 = o3d.pipelines.registration.registration_icp(
            p_pcd,
            gt_pcd,
            0.5,
            reg2.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-6, 50),
        )
        reconstructed_mesh.transform(reg3.transformation)

        reconstructed_pcd = reconstructed_mesh.sample_points_uniformly(number_of_points=100_000)
        reconstructed_pcd = torch.from_numpy(np.asarray(reconstructed_pcd.points)).cuda()
        gt_pcd = torch.from_numpy(np.asarray(gt_pcd.points)).cuda()
        cd = chamfer_distance(reconstructed_pcd, gt_pcd)
        print(f"Chamfer distance: {cd}")
        