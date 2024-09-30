import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, 
               app_model=None):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()
        out = render(view, gaussians, pipeline, background, app_model=app_model)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()
        depth = depth.detach().cpu().numpy()
        print(depth.min(), " ", depth.max())
        depth = -depth.clip(0, 4)
        depth = np.abs(depth)
        depth_i = (depth - depth.min()) / ( depth.max()- depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)

        normal = out["rendered_normal"].permute(1,2,0)
        normal = normal/(normal.norm(dim=-1, keepdim=True)+1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal+1) * 127.5).astype(np.uint8).clip(0, 255)

        if name == 'test':
            torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        else:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_i)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(model.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background)
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    