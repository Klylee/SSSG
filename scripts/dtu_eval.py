import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
import multiprocessing as mp
import argparse, os
import cv2

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

def chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    '''
    return: dual chamfer distance, points1 to points2, points2 to points1

    '''
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(points1)
    dist_d2s, idx_d2s = nn_engine.kneighbors(points2, n_neighbors=1, return_distance=True) 
    max_dist = args.max_dist 
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    nn_engine.fit(points2)
    dist_s2d, idx_s2d = nn_engine.kneighbors(points1, n_neighbors=1, return_distance=True) 
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
    return (mean_d2s + mean_s2d) / 2, mean_s2d, mean_d2s, dist_d2s

def depth_filter(points, depth_map, cameras):
    visibility = np.zeros(points.shape[0], dtype=bool)
    for camera in tqdm(cameras, desc="Culling mesh given masks"):      
        with torch.no_grad():
            # transform and project
            cam_points =  points @ camera.full_proj_transform
            ndc_points = cam_points[:, :2] / (cam_points[:, 2].unsqueeze(1) + 1e-6)
            point2d_x = (ndc_points[:, 0] + 1) * 0.5 * camera.image_width
            point2d_y = (ndc_points[:, 1] + 1) * 0.5 * camera.image_height
            point2d = torch.stack([point2d_x, point2d_y, cam_points[:, 2]], dim=-1)

            in_image = (point2d[:, 0] >= 0) & (point2d[:, 0] < camera.image_width) & (point2d[:, 1] >= 0) & (point2d[:, 1] < camera.image_height)
            point2d[:, 0] = point2d[:, 0].clamp(0, camera.image_width - 1)
            point2d[:, 1] = point2d[:, 1].clamp(0, camera.image_height - 1)

            image_id = int(camera.image_name.split('.')[0])
            depth_map = depths[image_id]
            depth = depth_map[point2d[:, 1].long(), point2d[:, 0].long()]
            
            valid = in_image & (point2d[:, 2] <= depth)
            visibility |= valid.cpu().numpy()

    return visibility

if __name__ == '__main__':
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    parser.add_argument('--data', type=str, default='data_in.ply')
    parser.add_argument('--scan', type=int, default=1)
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])
    parser.add_argument('--vis_out_dir', type=str, default='.')
    parser.add_argument('--downsample_density', type=float, default=0.2)
    parser.add_argument('--patch_size', type=float, default=60)
    parser.add_argument('--max_dist', type=float, default=20)
    parser.add_argument('--visualize_threshold', type=float, default=10)
    parser.add_argument('--s', type=str, default='chinesedragon')
    parser.add_argument('--dataset', type=str, default='/home/yuanyouwen/expdata/neuralto')
    parser.add_argument('--recon_f', type=str, default='output_neuralto/chinesedragon/test/mesh/tsdf_fusion_post.obj')
    parser.add_argument('--mc', action='store_true', default=False)
    args = parser.parse_args()

    dataset_dir = os.path.join(args.dataset, args.s)
    gt_mesh_file = os.path.join(args.dataset, 'gt_mesh', f'{args.s}.obj')
    reconstructed_mesh_file = args.recon_f
    reconstructed_mesh_dir = os.path.dirname(reconstructed_mesh_file)

    if not os.path.exists(gt_mesh_file):
        print('not exists gt mesh file: ' + gt_mesh_file)
        exit(0)

    if not os.path.exists(reconstructed_mesh_file):
        reconstructed_mesh_file = reconstructed_mesh_file.replace('.ply', '.obj')
        if not os.path.exists(reconstructed_mesh_file):
            print('not exists reconstructed mesh file: ' + reconstructed_mesh_file)
            exit(0)
    
    os.makedirs(os.path.join(reconstructed_mesh_dir, 'eval'), exist_ok=True)
    
    num_points = 1000_000

    mask_cull = args.mc
    
    print('')
    
    thresh = args.downsample_density
    if args.mode == 'mesh':
        reconstructed_mesh = trimesh.load_mesh(reconstructed_mesh_file)

        if mask_cull:
             # load masks
            train_cam_infos = readCamerasFromTransforms(dataset_dir, "transforms_train.json", white_background=False)
            test_cam_infos = readCamerasFromTransforms(dataset_dir, "transforms_test.json", white_background=False)
            train_cameras = cameraList_from_camInfos(train_cam_infos, 1.0, lp.extract(args))
            test_cameras = cameraList_from_camInfos(test_cam_infos, 1.0, lp.extract(args))
            all_cameras = train_cameras + test_cameras

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

                    point_valid = point2d[valid]
                    image = torch.zeros([3, camera.image_height, camera.image_width], dtype=torch.uint8)
                    mask_bg = torch.zeros_like(image)
                    mask_bg[0, :, :] = mask.float() * 255
                    image = image + mask_bg
                    image[0, point_valid[:, 1].long(), point_valid[:, 0].long()] = 255
                    image[1, point_valid[:, 1].long(), point_valid[:, 0].long()] = 255
                    image[2, point_valid[:, 1].long(), point_valid[:, 0].long()] = 255
                    image = image.clamp(0, 255).permute(1, 2, 0)
                    cv2.imwrite(f"debug/{camera.image_name}.png", image.cpu().numpy())

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
            o3d.io.write_triangle_mesh(os.path.join(reconstructed_mesh_dir, 'filtered_' + os.path.basename(reconstructed_mesh_file)), reconstructed_mesh)
            
        
        # align points cloud
        p_pcd = o3d.geometry.PointCloud()
        if mask_cull:
            p_pcd.points = o3d.utility.Vector3dVector(new_vertices)
        else:
            faces = reconstructed_mesh.faces
            vertices = reconstructed_mesh.vertices
            reconstructed_mesh = o3d.geometry.TriangleMesh()
            reconstructed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            reconstructed_mesh.triangles = o3d.utility.Vector3iVector(faces)
            p_pcd.points = o3d.utility.Vector3dVector(vertices)
        
        if Path(gt_mesh_file).suffix == '.ply':
            gt_pcd = o3d.io.read_point_cloud(gt_mesh_file)
        elif Path(gt_mesh_file).suffix == '.obj':
            tri_mesh = trimesh.load_mesh(gt_mesh_file)
            gt_mesh = o3d.geometry.TriangleMesh()
            gt_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
            gt_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
            gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=num_points)
        
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
        reconstructed_pcd = reconstructed_mesh.sample_points_uniformly(number_of_points=num_points)

        depth_map_file = f"/home/yuanyouwen/expdata/neuralto/depth/{args.s}.npy"
        if False and os.path.exists(depth_map_file):
            # load masks
            train_cam_infos = readCamerasFromTransforms(dataset_dir, "transforms_train.json", white_background=False)
            test_cam_infos = readCamerasFromTransforms(dataset_dir, "transforms_test.json", white_background=False)
            train_cameras = cameraList_from_camInfos(train_cam_infos, 1.0, lp.extract(args))
            all_cameras = train_cameras
            
            recon_points = torch.from_numpy(np.asarray(reconstructed_pcd.points)).cuda()
            recon_points = torch.cat((recon_points, torch.ones_like(recon_points[:, :1])), dim=-1).float()
            gt_points = torch.from_numpy(np.asarray(gt_pcd.points)).cuda()
            gt_points = torch.cat((gt_points, torch.ones_like(recon_points[:, :1])), dim=-1).float()

            depths = np.load(depth_map_file)
            depths = torch.from_numpy(depths).cuda()
            
            visibility = depth_filter(recon_points, depths, all_cameras)
            reconstructed_pcd.points = o3d.utility.Vector3dVector(np.asarray(reconstructed_pcd.points)[visibility])
            visibility = depth_filter(gt_points, depths, all_cameras)
            gt_pcd.points = o3d.utility.Vector3dVector(np.asarray(gt_pcd.points)[visibility])

        o3d.io.write_point_cloud(os.path.join(reconstructed_mesh_dir, 'eval', 'rec.ply'), reconstructed_pcd)
        o3d.io.write_point_cloud(os.path.join(reconstructed_mesh_dir, 'eval', 'gt.ply'), gt_pcd)

        cd, cd1, cd2, dist_rg = chamfer_distance(np.asarray(gt_pcd.points), np.asarray(reconstructed_pcd.points))
        
        print("-----------------------")
        print(reconstructed_mesh_file)
        print("-----------------------")
        print(f"Chamfer distance: {cd}")
        print(f"     gt -> recon: {cd1}")
        print(f"     recon -> gt: {cd2}")

        dist_degree = dist_rg.squeeze() / 0.002
        dist_degree = dist_degree.astype(int)
        dist_degree = np.clip(dist_degree, a_min=0, a_max=4)
        
        palette = np.array([
            [202, 243, 255],  # '#CAF3FFFF' 0.000 - 0.002
            [109, 213, 233],  # '#6DD5E9FF' 0.002 - 0.004
            [58, 154, 199],   # '#3A9AC7FF' 0.004 - 0.006
            [30, 107, 146],   # '#1E6B92FF' 0.006 - 0.008
            [20, 76, 102]     # '#144C66'   0.008 -
        ], dtype=np.float32) / 255.0
        colors = palette[dist_degree]
        
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(reconstructed_pcd.points)
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(reconstructed_mesh_dir, 'eval', f"colored-{args.s}.ply"), colored_pcd)