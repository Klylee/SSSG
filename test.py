from torchvision import utils
import torch
import torch.nn.functional as F
from scene import Scene, GaussianModel
import open3d as o3d

gaussians = GaussianModel(3)
gaussians.load_ply('./output_neuralto/yuanbao/test/point_cloud/iteration_80000/point_cloud.ply')
opacity = gaussians.get_opacity

opacity_filter = (opacity.squeeze()) > 0.5
centers = gaussians.get_xyz
centers = centers[opacity_filter]

point_cloud = o3d.geometry.PointCloud()

# 设置点云的点和颜色
point_cloud.points = o3d.utility.Vector3dVector(centers.tolist())  # 设置点坐标
point_cloud.colors = o3d.utility.Vector3dVector(torch.zeros_like(centers).tolist())  # 设置点颜色

# 保存点云到文件，例如保存为 PLY 格式
o3d.io.write_point_cloud("output.ply", point_cloud)