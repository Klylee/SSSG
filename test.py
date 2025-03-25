from torchvision import utils
import torch
import torch.nn.functional as F
from scene import Scene, GaussianModel
import open3d as o3d
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
import sys
from scene.direct_light_map import DirectLightMap

parser = ArgumentParser(description="Training script parameters")
lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
parser.add_argument('--ip', type=str, default="127.0.0.1")
parser.add_argument('--port', type=int, default=6007)
parser.add_argument('--debug_from', type=int, default=-100)
parser.add_argument('--detect_anomaly', action='store_true', default=False)
parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 50_000, 80_000])
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000, 50_000, 80_000])
parser.add_argument("--start_checkpoint", type=str, default = None)
args = parser.parse_args(sys.argv[1:])
args.save_iterations.append(args.iterations)

print("Optimizing " + args.model_path)

# gaussians = GaussianModel(3)
# checkpoint = "./output_neuralto/chinesedragon/test/inner_chkpnt80000.pth"
# (model_params, first_iter) = torch.load(checkpoint)
# gaussians.restore(model_params, op.extract(args))

# # gaussians.load_ply('./output_neuralto/yuanbao/test/point_cloud/iteration_80000/point_cloud.ply')
# opacity = gaussians.get_opacity

# opacity_filter = (opacity.squeeze()) > 0.1
# centers = gaussians.get_xyz
# centers = centers[opacity_filter]

# point_cloud = o3d.geometry.PointCloud()

# # 设置点云的点和颜色
# point_cloud.points = o3d.utility.Vector3dVector(centers.tolist())  # 设置点坐标
# point_cloud.colors = o3d.utility.Vector3dVector(torch.zeros_like(centers).tolist())  # 设置点颜色

# # 保存点云到文件，例如保存为 PLY 格式
# o3d.io.write_point_cloud("output.ply", point_cloud)

checkpoint = "./output_neuralto/chinesedragon/test/env_light80000.pth"
direct_light = DirectLightMap()
direct_light.create_from_ckpt(checkpoint, op.extract(args))
