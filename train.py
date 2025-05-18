import os
import time
import torch
import random
import numpy as np
import cv2
import uuid
import sys, time
import torch.nn.functional as F
from datetime import datetime
from random import randint
from gaussian_renderer import render, network_gui
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from scene.app_model import AppModel
from scene.cameras import Camera
from scene.direct_light_map import DirectLightMap
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight, first_order_edge_aware_loss
from utils.graphics_utils import patch_offsets, patch_warp
from utils.general_utils import safe_state, build_scaling_rotation
from utils.image_utils import psnr, erode
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(22)

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0

    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)
    
    os.system(f'rm -rf {dataset.model_path}/debug')
    # os.system(f'rm -rf {dataset.model_path}/app_model')

    gaussians = GaussianModel(dataset.sh_degree)
    inner_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, inner_gaussians)
    gaussians.training_setup(opt)
    inner_gaussians.training_setup(opt)

    if gaussians.use_pbr:
        direct_light = DirectLightMap()
        direct_light.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    
    scene_name = dataset.model_path.split('/')[-2]
    checkpoint = f"./output_neuralto/{scene_name}/test/chkpnt30000.pth"
    if checkpoint and os.path.exists(checkpoint):
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

        if os.path.exists(checkpoint.replace('chkpnt', 'inner_chkpnt')):
            (model_params, first_iter) = torch.load(checkpoint.replace('chkpnt', 'inner_chkpnt'))
            inner_gaussians.restore(model_params, opt)

        if gaussians.use_pbr and os.path.exists(checkpoint.replace('chkpnt', 'env_light')):
            direct_light.create_from_ckpt(checkpoint.replace('chkpnt', 'env_light'), opt, True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    normal_loss, geo_loss, ncc_loss = None, None, None
    loss_pbr = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    if gaussians.use_pbr:
        gaussians.update_visibility(pipe.sample_num)

    use_inner_gs = True
    inner_gs_start = 1000
    ratio = 1.0
    opt.multi_view_weight_from_iter = 75000
    opt.densify_until_iter = 60000
    reset_opacity_until_iter = 60000

    camera = scene.getTrainCameras()[0]
    print(f'{camera.resolution} {camera.ncc_scale}')

    use_density_prune = True
    for iteration in range(first_iter, opt.iterations + 1):
        # if iteration > 60000:
        #     use_density_prune = False
        
        iter_start.record()
        xyz_lr = gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # if iteration > opt.single_view_weight_from_iter and gaussians.get_xyz.shape[0] != 0 and gaussians.get_xyz.shape[0] != gaussians._incident_dirs.shape[0]:
        #     gaussians.update_visibility(pipe.sample_num)

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>opt.single_view_weight_from_iter, 
                            return_depth_normal=iteration>opt.single_view_weight_from_iter,
                            return_pbr=True, direct_light=direct_light, inner_gs=inner_gaussians if use_inner_gs else None)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if use_inner_gs and iteration > opt.single_view_weight_from_iter + inner_gs_start:
            image = render_pkg['render1']

        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        loss = image_loss.clone()

        # <start> 2025-03-12
        # restrain the inner_gs to the inner part of the model
        # if use_inner_gs and use_density_prune == False:
        if False and use_inner_gs and iteration > 50000:
            inner_gs_points = torch.cat([inner_gaussians.get_xyz, torch.ones([inner_gaussians.get_xyz.shape[0], 1], device="cuda")], dim=-1)
            cam_points =  inner_gs_points @ viewpoint_cam.full_proj_transform
            ndc_points = cam_points[:, :2] / (cam_points[:, 2].unsqueeze(1) + 1e-6)
            point2d_x = (ndc_points[:, 0] + 1) * 0.5 * viewpoint_cam.image_width
            point2d_y = (ndc_points[:, 1] + 1) * 0.5 * viewpoint_cam.image_height
            point2d = torch.stack([point2d_x, point2d_y, cam_points[:, 2]], dim=-1)
            in_image = (point2d[:, 0] >= 0) & (point2d[:, 0] < viewpoint_cam.image_width) & (point2d[:, 1] >= 0) & (point2d[:, 1] < viewpoint_cam.image_height)
            point2d = point2d[in_image]
            point2d_x = point2d[:, 0].clamp(0, viewpoint_cam.image_width - 1).long()
            point2d_y = point2d[:, 1].clamp(0, viewpoint_cam.image_height - 1).long()
            surface_depth = render_pkg['plane_depth'].squeeze()[point2d_y, point2d_x]
            depth_diff = surface_depth - point2d[:, 2]
            
            inner_opacity = inner_gaussians.get_opacity
            inner_opacity = inner_opacity[in_image]
            inner_opacity = inner_opacity[depth_diff > 0 - 0.00001]
            inner_restrain_loss = 0.01 * depth_diff[depth_diff > 0 - 0.00001].mean()
            loss += inner_restrain_loss

            # ratio = inner_opacity.shape[0] / inner_gaussians.get_opacity.shape[0]
        # <end>
        # ratio = (inner_gaussians.get_sign < 0).sum() / inner_gaussians.get_sign.shape[0]
        

        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()
        # single-view loss
        if iteration > opt.single_view_weight_from_iter:
            # mask loss
            mask = viewpoint_cam.mask.float()
            loss_mask = l1_loss(render_pkg['alpha'], mask)
            loss += 0.05 * loss_mask + 0.9 * min((iteration - opt.single_view_weight_from_iter) / (80000 - opt.single_view_weight_from_iter), 1.0)

            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            # normal = F.normalize(normal, dim=0)
            # depth_normal = F.normalize(depth_normal, dim=0)

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2

            # normal_loss = weight * (1 - (F.cosine_similarity(depth_normal, normal, dim=0)).mean())
            single_view_weight = opt.single_view_weight + 0.1 * min((iteration - opt.single_view_weight_from_iter) / (80000 - opt.single_view_weight_from_iter), 1.0)
            normal_loss = single_view_weight * ((((depth_normal - normal)).abs().sum(0))).mean()
            loss += normal_loss
                    
            

            if gaussians.use_pbr:
                rendered_pbr = render_pkg["pbr"]
                rendered_base_color = render_pkg["base_color"]
                rendered_roughness = render_pkg["roughness"]
                rendered_scatter = render_pkg["scatter"]
                rendered_specular = render_pkg["specular"]

                Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
                ssim_val_pbr = ssim(rendered_pbr, gt_image)
                loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
                
                loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * mask, gt_image)
                loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness.unsqueeze(0) * mask, gt_image)

                weight = 1.1
                if iteration > opt.single_view_weight_from_iter + inner_gs_start + 1000:
                    loss += weight * loss_pbr
                    loss += 0.01 * loss_base_color_smooth
                    loss += 0.01 * loss_roughness_smooth


        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                
                d_mask = d_mask & (pixel_noise < pixel_noise_th)
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0
                
                
                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # visualization
            if iteration > opt.single_view_weight_from_iter and iteration % 200 == 0:
                gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                if 'app_image' in render_pkg:
                    img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                else:
                    img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)

                # if iteration > opt.multi_view_weight_from_iter:
                #     d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                #     d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                # else:
                # d_mask_show_color = np.zeros_like(gt_img_show).astype(np.uint8)
                
                depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                image_weight = image_weight.detach().cpu().numpy()
                image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                specular_show = ((rendered_specular).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                pbr_show = ((rendered_pbr).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                scatter_show = ((rendered_scatter).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                alpha_show = (render_pkg["alpha"].repeat(3,1,1).permute(1,2,0).clamp(0,1) * 255).detach().cpu().numpy().astype(np.uint8)
                row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color, specular_show], axis=1)
                row1 = np.concatenate([alpha_show, depth_color, depth_normal_show, scatter_show, pbr_show], axis=1)
                image_to_show = np.concatenate([row0, row1], axis=0)
                cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)
            
            
            # if iteration < opt.single_view_weight_from_iter and iteration % 500 == 0:
            #     gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            #     img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            #     res = np.concatenate([gt_img_show, img_show], axis=1)
            #     cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), res)

            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            ema_pbr_loss_for_log = 0.4 * loss_pbr.item() if loss_pbr is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Single": f"{ema_single_view_for_log:.{5}f}",
                    "pbr": f"{ema_pbr_loss_for_log:.{5}f}",
                    "light": f"{direct_light.colocated_light.item():.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "inner": f"{len(inner_gaussians.get_xyz)}",
                    "ratio": f"{ratio:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # <start> modify 2025-03-17
            # Densification
            if use_density_prune:
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    mask = (render_pkg["out_observe"] > 0) & visibility_filter
                    gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                    gaussians.add_densification_stats(viewspace_point_tensor, render_pkg["viewspace_points_abs"], visibility_filter)

                    if use_inner_gs and iteration > opt.single_view_weight_from_iter + inner_gs_start:
                        inner_mask = (render_pkg["inner_observe"] > 0) & render_pkg["inner_vis_filter"]
                        inner_gaussians.max_radii2D[inner_mask] = torch.max(inner_gaussians.max_radii2D[inner_mask], render_pkg["inner_radii"][inner_mask])
                        inner_gaussians.add_densification_stats(render_pkg["inner_viewspace_points"], render_pkg["inner_viewspace_points_abs"], render_pkg["inner_vis_filter"])

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
                        
                        # if use_inner_gs and iteration > opt.single_view_weight_from_iter + inner_gs_start:
                        #     dead_mask = ((inner_gaussians.get_opacity).abs() <= 0.005).squeeze(-1)
                        #     inner_gaussians.relocate_gs(dead_mask=dead_mask)
                        #     inner_gaussians.add_new_gs(cap_max=20000)

                        if use_inner_gs and iteration > opt.single_view_weight_from_iter + inner_gs_start:
                            inner_gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
            
                # multi-view observe trim
                if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                    observe_the = 2
                    observe_cnt = torch.zeros_like(gaussians.get_opacity)
                    for view in scene.getTrainCameras():
                        render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                        out_observe = render_pkg_tmp["out_observe"]
                        observe_cnt[out_observe > 0] += 1
                    prune_mask = (observe_cnt < observe_the).squeeze()
                    if prune_mask.sum() > 0:
                        gaussians.prune_points(prune_mask)

                # reset_opacity
                if iteration < reset_opacity_until_iter:
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                        if use_inner_gs and iteration > opt.single_view_weight_from_iter + inner_gs_start:
                            inner_gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    app_model.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    app_model.optimizer.zero_grad(set_to_none = True)

                    if use_inner_gs and iteration > opt.single_view_weight_from_iter + inner_gs_start:
                        inner_gaussians.optimizer.step()
                        inner_gaussians.optimizer.zero_grad(set_to_none = True)

                        inner_gaussians.add_noise(xyz_lr)

                    if gaussians.use_pbr:
                        direct_light.step()

            else:
                if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    dead_mask = (gaussians.get_opacity.abs() <= 0.005).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask)
                    gaussians.add_new_gs(cap_max=100000)

                    if use_inner_gs and iteration > opt.single_view_weight_from_iter + inner_gs_start:
                        dead_mask = (inner_gaussians.get_opacity <= 0.005).squeeze(-1)
                        inner_gaussians.relocate_gs(dead_mask=dead_mask)
                        inner_gaussians.add_new_gs(cap_max=-1)
                
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    app_model.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    app_model.optimizer.zero_grad(set_to_none = True)

                    gaussians.add_noise(xyz_lr)

                    if use_inner_gs and iteration > opt.single_view_weight_from_iter + inner_gs_start:
                        inner_gaussians.optimizer.step()
                        inner_gaussians.optimizer.zero_grad(set_to_none = True)

                        inner_gaussians.add_noise(xyz_lr)

                    if gaussians.use_pbr:
                        direct_light.step()
                    
            # <end>

            # if iteration % 100 == 0:
            #     with open('./debug/point_num.txt', '+a') as f:
            #         ratio = (inner_gaussians.get_opacity < 0).sum() / inner_gaussians.get_xyz.shape[0] + 1e-8 
            #         f.write(f"{gaussians.get_xyz.shape[0]}, {inner_gaussians.get_xyz.shape[0]}, {ratio}\n")
            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)
                torch.save((direct_light.capture(), iteration), scene.model_path + '/env_light' + str(iteration) + '.pth')
                torch.save((inner_gaussians.capture(), iteration), scene.model_path + "/inner_chkpnt" + str(iteration) + ".pth")
            # if iteration == 33000:
            #     torch.save((inner_gaussians.capture(), iteration), scene.model_path + "/inner_chkpnt" + str(iteration) + ".pth")
    
    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000, 50_000, 70000, 80_000, 90_000, 100_000, 110_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000, 50_000, 70000, 80_000, 90_000, 100_000, 110_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")