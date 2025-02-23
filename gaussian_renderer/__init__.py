#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.app_model import AppModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image, rgb_to_srgb, fibonacci_sphere_sampling
from utils.loss_utils import ssim
from utils.image_utils import psnr

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           app_model: AppModel=None, return_plane = True, return_depth_normal = True, return_pbr = False, direct_light = None, inner_gs = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return_dict = None
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy, 
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    

    

    if not return_plane:
        rendered_image, radii, out_observe, out_all_map, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        if pc.use_pbr and return_pbr:
            rendered_pbr = out_all_map[5:8, ]
            rendered_base_color = out_all_map[8:11, ]
            rendered_roughness = out_all_map[11, ]
            rendered_diffuse = out_all_map[12:15, ]
            rendered_visibility = out_all_map[15:, ]
            
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe}
        
        if app_model is not None and pc.use_app:
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
            app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
            return_dict.update({"app_image": app_image})
        return return_dict


    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    # depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 30)).cuda().float()

    input_all_map[:, :3] = F.normalize(local_normal)
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance
    
    F0=None
    if pc.use_pbr and return_pbr:
        base_color = pc.get_base_color
        roughness = pc.get_roughness
        viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
        incidents = pc.get_incidents

        global_normal_p = pc.get_normal_p(viewpoint_camera)

        if pc._incident_dirs.shape[0] == incidents.shape[0]:
            brdf_color, extra_results = rendering_equation(
                base_color, roughness, global_normal, viewdirs, incidents, pc.get_scattersColor,
                direct_light, visibility_precompute=pc._visibility_tracing, incident_dirs_precompute=pc._incident_dirs, incident_areas_precompute=pc._incident_areas, world_to_view=viewpoint_camera.world_view_transform[:3,:3])
        
            input_all_map[:, 5:8] = brdf_color
            input_all_map[:, 8:11] = base_color
            input_all_map[:, 11] = roughness.squeeze()
            input_all_map[:, 12:15] = extra_results["diffuse_light"]
            input_all_map[:, 15:18] = F.normalize(global_normal_p @ viewpoint_camera.world_view_transform[:3,:3]) # property normal

            input_all_map[:, 18:21] = extra_results["specular"]
            # input_all_map[:, 21] = extra_results["opacity_in"]

            F0 = extra_results['F0']

            if inner_gs is not None:
                inner_means3D = inner_gs.get_xyz
                inner_means2D = torch.zeros_like(inner_gs.get_xyz, dtype=inner_gs.get_xyz.dtype, requires_grad=True, device="cuda") + 0
                inner_means2D_abs = torch.zeros_like(inner_gs.get_xyz, dtype=inner_gs.get_xyz.dtype, requires_grad=True, device="cuda") + 0
                try:
                    inner_means2D.retain_grad()
                    inner_means2D_abs.retain_grad()
                except:
                    pass
                inner_opacity = inner_gs.get_opacity
                inner_scales = inner_gs.get_scaling
                inner_rotations = inner_gs.get_rotation
                inner_shs = inner_gs.get_features

                means3D_all = torch.cat([means3D, inner_means3D], dim=0)
                means2D_all = torch.cat([means2D, inner_means2D], dim=0)
                means2D_abs_all = torch.cat([means2D_abs, inner_means2D_abs], dim=0)
                shs_all = torch.cat([shs, inner_shs], dim=0)
                opacity_all = torch.cat([opacity * F0, inner_opacity], dim=0)
                scales_all = torch.cat([scales, inner_scales], dim=0)
                rotations_all = torch.cat([rotations, inner_rotations], dim=0)

                # cat input_all_map
                inner_input_all_map = torch.zeros((inner_means3D.shape[0], 30)).cuda().float()

                inner_viewdirs = F.normalize(viewpoint_camera.camera_center - inner_means3D, dim=-1)
                inner_colors = eval_sh(inner_gs.active_sh_degree, inner_gs.get_features.transpose(1,2).view(-1, 3, (inner_gs.max_sh_degree + 1)**2), inner_viewdirs)
                inner_input_all_map[:, 5:8] = torch.clamp_min(inner_colors + 0.5, 0.0)

                input_all_map_all = torch.cat([input_all_map, inner_input_all_map], dim=0)
            else:
                means3D_all = means3D
                means2D_all = means2D
                means2D_abs_all = means2D_abs
                shs_all = shs
                opacity_all = opacity * F0
                scales_all = scales
                rotations_all = rotations
                input_all_map_all = input_all_map

            rendered_image1, _radii, _observe, out, _ = rasterizer(
            means3D = means3D_all,
            means2D = means2D_all,
            means2D_abs = means2D_abs_all,
            shs = shs_all,
            colors_precomp = colors_precomp,
            opacities = opacity_all,
            scales = scales_all,
            rotations = rotations_all,
            all_map = input_all_map_all,
            cov3D_precomp = cov3D_precomp)

            rendered_alpha1 = out[3, ]
            rendered_pbr = out[5:8, ]

            if inner_gs is not None:
                inner_radii = _radii[means3D.shape[0]:]
                inner_observe = _observe[means3D.shape[0]:]
    
    
    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]

    if pc.use_pbr and return_pbr:
        # rendered_pbr = out_all_map[5:8, ]
        rendered_base_color = out_all_map[8:11, ]
        rendered_roughness  = out_all_map[11, ]
        rendered_diffuse    = out_all_map[12:15, ]
        rendered_normal_p   = out_all_map[15:18, ]
        rendered_specular   = out_all_map[18:21, ]
        rendered_opacity_in = out_all_map[21, ]

        # add 2025-2-19 <start>
        # 在屏幕空间计算specular
        normal_dot_view = torch.sum(rendered_normal.permute(1,2,0) * viewpoint_camera.get_rays(), dim=-1) # [H, W]
        # normal_dot_view = normal_dot_view.clamp(min=0, max=1)
        # tan_theta = torch.sqrt(1.0 - normal_dot_view * normal_dot_view) / (normal_dot_view + 1e-10)
        # root = rendered_roughness * tan_theta  
        # G = 2.0 / (1.0 + torch.hypot(root, torch.ones_like(root))) 
        # dot_2 = normal_dot_view * normal_dot_view
        # root_2 = dot_2 + (1.0 - dot_2) / (rendered_roughness * rendered_roughness + 1e-10)
        # D = 1.0 / (math.pi * rendered_roughness * rendered_roughness * root_2 * root_2 + 1e-10)
        # light_color = direct_light.colocated_light.unsqueeze(0).unsqueeze(0).repeat(normal_dot_view.shape[0], normal_dot_view.shape[1], 3)
        # specular_color = 0.5 * light_color * 0.04 * G.unsqueeze(-1) * D.unsqueeze(-1) / (4.0 * normal_dot_view.unsqueeze(-1) + 1e-10) # [H, W, 3]
        # specular_color = specular_color.permute(2, 0, 1)

        # try blinn-phong model 2025-2-19
        specular_intensity = 0.5 * torch.pow(normal_dot_view, 32)
        light_color = direct_light.colocated_light.unsqueeze(0).unsqueeze(0).repeat(normal_dot_view.shape[0], normal_dot_view.shape[1], 3)
        specular_color = light_color * specular_intensity.unsqueeze(-1)
        specular_color = specular_color.permute(2, 0, 1)
        # <end>

    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rendered_normal": rendered_normal,
                    "plane_depth": plane_depth,
                    "rendered_distance": rendered_distance,
                    }
    
    if pc.use_pbr and return_pbr and pc._incident_dirs.shape[0] == incidents.shape[0]:
        return_dict.update({"base_color": rgb_to_srgb(rendered_base_color),
                            "render1": rendered_image1,
                            "alpha": rendered_alpha,
                            "pbr": rgb_to_srgb(rendered_pbr),
                            "rendered_normal_p": rendered_normal_p,
                            "roughness": rendered_roughness,
                            "diffuse": rgb_to_srgb(rendered_diffuse),
                            "specular": specular_color,
                        })
        if inner_gs is not None:
            return_dict.update({
                "inner_viewspace_points": inner_means2D,
                "inner_viewspace_points_abs": inner_means2D_abs,
                "inner_radii": inner_radii,
                "inner_vis_filter": inner_radii > 0,
                "inner_observe": inner_observe
            })
    
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image})   

    if return_depth_normal:
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict

def calculate_loss(viewpoint_camera, pc, results, opt, direct_light_env_light):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    rendered_image = results["render"]
    rendered_depth = results["depth"]
    rendered_normal = results["rendered_normal"]
    rendered_pbr = results["pbr"]
    rendered_opacity = results["opacity"]
    rendered_base_color = results["base_color"]
    rendered_roughness = results["roughness"]
    rendered_diffuse = results["diffuse"]

    gt_image = viewpoint_camera.original_image.cuda()
    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
    ssim_val_pbr = ssim(rendered_pbr, gt_image)
    tb_dict["l1_pbr"] = Ll1_pbr.item()
    tb_dict["ssim_pbr"] = ssim_val_pbr.item()
    tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
    loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
    loss = loss + opt.lambda_pbr * loss_pbr

    if opt.lambda_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        image_mask = viewpoint_camera.image_mask.cuda().bool()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask, depth_mask)

        loss_depth = F.l1_loss(rendered_depth[~sur_mask], gt_depth[~sur_mask])
        tb_dict["loss_depth"] = loss_depth.item()
        loss = loss + opt.lambda_depth * loss_depth

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = results['depth_normal']
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    if opt.lambda_normal_mvs_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        depth_mask = (gt_depth > 0).float()
        mvs_normal = viewpoint_camera.normal.cuda()
        loss_normal_mvs_depth = F.mse_loss(
            rendered_normal * depth_mask, mvs_normal * depth_mask)
        tb_dict["loss_normal_mvs_depth"] = loss_normal_mvs_depth.item()
        loss = loss + opt.lambda_normal_mvs_depth * loss_normal_mvs_depth

    if opt.lambda_light > 0:
        diffuse_light = results["diffuse_light"]
        mean_light = diffuse_light.mean(-1, keepdim=True).expand_as(diffuse_light)
        loss_light = F.l1_loss(diffuse_light, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    # if opt.lambda_base_color_smooth > 0:
    #     image_mask = viewpoint_camera.image_mask.cuda()
    #     loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
    #     # loss_base_color_smooth = second_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
    #     tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
    #     loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

    # if opt.lambda_roughness_smooth > 0:
    #     image_mask = viewpoint_camera.image_mask.cuda()
    #     loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
    #     # loss_roughness_smooth = second_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
    #     tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
    #     loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
    
    
    # if opt.lambda_light_smooth > 0:
    #     image_mask = viewpoint_camera.image_mask.cuda()
    #     loss_light_smooth = first_order_edge_aware_loss(rendered_diffuse * image_mask, rendered_normal)
    #     # loss_light_smooth = second_order_edge_aware_loss(rendered_diffuse * image_mask, gt_image)
    #     tb_dict["loss_light_smooth"] = loss_light_smooth.item()
    #     loss = loss + opt.lambda_light_smooth * loss_light_smooth
        
    # if opt.lambda_env_smooth > 0:
    #     env = direct_light_env_light.get_env
    #     loss_env_smooth = tv_loss(env[0].permute(2, 0, 1))
    #     tb_dict["loss_env_smooth"] = loss_env_smooth.item()
    #     loss = loss + opt.lambda_env_smooth * loss_env_smooth
    
    # if opt.lambda_normal_smooth > 0:
    #     # loss_normal_smooth = second_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
    #     loss_normal_smooth = tv_loss(rendered_normal * image_mask)
    #     tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
    #     loss = loss + opt.lambda_normal_smooth * loss_normal_smooth
    
    tb_dict["loss"] = loss.item()

    return loss, tb_dict

def rendering_equation(base_color, roughness, normals, viewdirs,
                              incidents, scatters, direct_light=None,
                              visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None, world_to_view=None):
    incident_dirs, incident_areas = incident_dirs_precompute, incident_areas_precompute

    deg = int(math.sqrt(incidents.shape[1]) - 1)
    local_incident_lights = eval_sh(deg, incidents.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)

    global_incident_lights = direct_light.direct_light(torch.matmul(incident_dirs, world_to_view))

    viewdirDotIncidentdir = incident_dirs @ viewdirs.unsqueeze(2)
    # parallel_lights = direct_light_env_light.light_strength * viewdirDotIncidentdir
    # parallel_lights = parallel_lights.clamp(min=0).repeat(1,1,3)

    # global_incident_lights = global_incident_lights * visibility_precompute
    incident_lights = local_incident_lights + global_incident_lights # [N, Sample, 3]
    # incident_lights = local_incident_lights + parallel_lights
    # viewdirs: [N, 3]
    # incident_dirs: [N, Samples, 3]
    
    normal_dot_view = (normals * viewdirs).sum(-1).unsqueeze(1) # [N,1]
    normal_dot_view = normal_dot_view.clamp(min=0, max=1)

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / math.pi
    f_s, fracRatio = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    deg = int(math.sqrt(scatters.shape[1]) - 1)
    scatter_colors = (eval_sh(deg, scatters.transpose(1,2).view(-1, 1, 3,(deg+1)**2), incident_dirs)).clamp_min(0) # [N,]
    scatter_colors = scatter_colors * (-viewdirDotIncidentdir).clamp(min=0) # [N, Samples, 3]
    scatter_color = scatter_colors.mean(dim=-2)
    
    F0 = 0.04 + 0.96 * (1 - normal_dot_view)

    # singleScatterColor = (base_color.unsqueeze(1).repeat(1, incident_dirs.shape[1], 1) * scatterColors) * (-viewdirDotIncidentdir.clamp(min=0)) # [N, Samples, 3]
    # singleScatterColor = base_color * opacity_in.unsqueeze(1) # [N,]

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    specular_color = ((f_s) * transport).mean(dim=-2)
    pbr = F0 * (((f_d + f_s) * transport).mean(dim=-2)) + (1 - F0) * scatter_color
    # pbr = ((f_d + f_s) * transport).mean(dim=-2)
    # diffuse_light = transport.mean(dim=-2)

    # add 2025-2-18 <start>
    trans_albedo = 0.8
    F = 0.04
    Fss90 = roughness
    Fss = (1 + (Fss90 - 1) * F) * (1 + (Fss90 - 1) * F)
    dot_abs = torch.abs(normal_dot_view)
    S = 1.25 * (Fss * (1 / (dot_abs + dot_abs) - 0.5) + 0.5)
    fr = S / math.pi
    surf_color = (1 - trans_albedo) * direct_light.colocated_light * base_color * fr * normal_dot_view

    # sin_theta = torch.sqrt(1.0 - normal_dot_view * normal_dot_view)
    # tan_theta = sin_theta / (normal_dot_view + 1e-10)
    # root = roughness * tan_theta  
    # G = 2.0 / (1.0 + torch.hypot(root, torch.ones_like(root))) 

    # dot_2 = normal_dot_view * normal_dot_view
    # root_2 = dot_2 + (1.0 - dot_2) / (roughness * roughness + 1e-10)
    # D = 1.0 / (math.pi * roughness * roughness * root_2 * root_2 + 1e-10)
    # specular_color = 0.5 * direct_light.colocated_light * F * G * D / (4.0 * normal_dot_view + 1e-10)

    # specular_intensity = 0.5 * torch.pow(normal_dot_view, 32)
    # light_color = direct_light.colocated_light.unsqueeze(0).repeat(normal_dot_view.shape[0], 3)
    # specular_color = light_color * specular_intensity

    # pbr = surf_color + specular_color
    # </end>

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        # "global_incident_lights": global_incident_lights,
        # "incident_visibility": visibility_precompute,
        "diffuse_light": scatter_color,
        "specular": specular_color,
        "F0": F0
    }

    return pbr, extra_results


def GGX_specular(normal, pts2c, pts2l, roughness, fresnel):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = (alpha * alpha).unsqueeze(1)  # [nrays, 1, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2 - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * math.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * math.pi)
    spec = frac / nom
    return spec, frac0
