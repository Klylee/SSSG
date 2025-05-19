import torch
import math
import torch.nn.functional as F
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.app_model import AppModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image, rgb_to_srgb

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
        rendered_image, radii, out_observe, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
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
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 15)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    if pc.use_pbr and return_pbr:
        base_color = pc.get_base_color
        roughness = pc.get_roughness
        viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

        normal_dot_view = (global_normal * viewdirs).sum(-1).unsqueeze(1) # [N,1]
        normal_dot_view = normal_dot_view.clamp(min=0, max=1)

        inner_colors = eval_sh(pc.active_sh_degree, pc.get_features.transpose(1,2).view(-1, 3, (pc.max_sh_degree + 1)**2), viewdirs)
        input_all_map[:, 5:8] = torch.clamp_min(inner_colors + 0.5, 0.0)
        input_all_map[:, 8:11] = base_color
        input_all_map[:, 11] = roughness.squeeze()

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
    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rendered_normal": rendered_normal,
                    "plane_depth": plane_depth,
                    "rendered_distance": rendered_distance,
                    "alpha": rendered_alpha
                    }
    depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
    if return_depth_normal:
        return_dict.update({"depth_normal": depth_normal})
    
    if pc.use_pbr and return_pbr:
        rendered_scatter = out_all_map[5:8, ]
        rendered_base_color = out_all_map[8:11, ]
        rendered_roughness  = out_all_map[11, ]

        fresnel = 0.04
        viewdirs = F.normalize(viewpoint_camera.get_rays(), dim=-1) # [H, W, 3]
        normal_map = F.normalize(rendered_normal.permute(1,2,0), dim=-1) # [H, W, 3]
        nov = torch.sum(normal_map * -viewdirs, dim=-1, keepdim=True).clamp(1e-6, 0.999999) # [H, W, 1]
        frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, ((-5.55473) * nov - 6.98316) * nov)
        alpha = (rendered_roughness * rendered_roughness).unsqueeze(2)
        alpha2 = alpha * alpha
        k = (alpha + 2 * rendered_roughness.unsqueeze(2) + 1.0) / 8.0
        nom0 = nov * nov * (alpha2 - 1) + 1
        nom1 = nov * (1 - k) + k
        f_s = frac0 * alpha2 / (4 * math.pi * nom0 * nom0 * nom1 * nom1).clamp_(1e-6, 4 * math.pi) # [H, W, 1]
        f_s = f_s.repeat(1, 1, 3).permute(2, 0, 1)
        f_d = rendered_base_color / math.pi
        light_colors = direct_light.colocated_light.unsqueeze(0).unsqueeze(0).repeat(1, nov.shape[0], nov.shape[1]) # [1, H, W]
        light_colors = light_colors / (1 + 0.7 * plane_depth + 1.8 * plane_depth * plane_depth)
        light_colors = light_colors.repeat(3, 1, 1)
        surf_color = light_colors * f_d
        specular = light_colors * f_s

        pbr = specular + 0.3 * (1-frac0.permute(2,0,1)) * (1-frac0.permute(2,0,1)) * surf_color  + (1 - frac0.permute(2,0,1)) * rendered_scatter

        return_dict.update(
            {"base_color": rgb_to_srgb(rendered_base_color),
            "pbr": rgb_to_srgb(pbr),
            "scatter": rendered_scatter,
            "roughness": rendered_roughness,
            "diffuse": rgb_to_srgb(surf_color),
            "specular": specular,
        })

        
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image})   

    
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict

def rendering_equation(base_color, roughness, normals, viewdirs, scatters, direct_light=None):
    fresnel = 0.04

    normal_dot_view = (normals * viewdirs).sum(-1).unsqueeze(1) # [N,1]
    normal_dot_view = normal_dot_view.clamp(min=0, max=1)
    
    roughness_2 = roughness * roughness
    roughness_4 = roughness_2 * roughness_2
    k = (roughness_2 + 2 * roughness + 1.0) / 8.0
    nom1 = normal_dot_view * (1 - k) + k
    nom0 = normal_dot_view * normal_dot_view * (roughness_4 - 1) + 1

    _D = math.pi * nom0 * nom0
    _G = nom1 * nom1
    
    F0 = fresnel + (1 - fresnel) * (1 - normal_dot_view)
    f_s = F0 * roughness_4 / (4 * _D * _G).clamp_(1e-6, 4 * math.pi)
    f_d = 0.3 * (1 - F0) * (1 - F0) * base_color / math.pi

    # f_s, fracRatio = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    deg = int(math.sqrt(scatters.shape[1]) - 1)
    scatter_color = (eval_sh(deg, scatters.transpose(1,2).view(-1, 3,(deg+1)**2), viewdirs)).clamp_min(0) # [N,]
    

    # singleScatterColor = (base_color.unsqueeze(1).repeat(1, incident_dirs.shape[1], 1) * scatterColors) * (-viewdirDotIncidentdir.clamp(min=0)) # [N, Samples, 3]
    # singleScatterColor = base_color * opacity_in.unsqueeze(1) # [N,]

    light_colors = direct_light.colocated_light.unsqueeze(0).repeat(normal_dot_view.shape[0], 3) # [N, 3]
    # transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    specular_color = f_s * light_colors
    # pbr = F0 * (((f_d + f_s) * transport).mean(dim=-2)) + (1 - F0) * scatter_color
    pbr = (f_d + f_s) * light_colors + (1 - F0) * scatter_color
    # pbr = ((f_d + f_s) * transport).mean(dim=-2)
    # diffuse_light = transport.mean(dim=-2)

    # add 2025-2-18 <start>
    # trans_albedo = 0.8
    # F = 0.04
    # Fss90 = roughness
    # Fss = (1 + (Fss90 - 1) * F) * (1 + (Fss90 - 1) * F)
    # dot_abs = torch.abs(normal_dot_view)
    # S = 1.25 * (Fss * (1 / (dot_abs + dot_abs) - 0.5) + 0.5)
    # fr = S / math.pi
    # surf_color = (1 - trans_albedo) * direct_light.colocated_light * base_color * fr * normal_dot_view

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
