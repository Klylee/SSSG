import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from pytorch3d.transforms import quaternion_to_matrix
# from pytorch3d.ops import knn_points
from bvh import RayTracer
from utils.graphics_utils import fibonacci_sphere_sampling
from utils.reloc_utils import compute_relocation_cuda

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = torch.nn.functional.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = torch.nn.functional.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        if self.use_pbr:
            self.base_color_activation = lambda x: torch.sigmoid(x) * 0.77 + 0.03
            self.roughness_activation = lambda x: torch.sigmoid(x) * 0.9 + 0.09
            self.inverse_roughness_activation = lambda y: inverse_sigmoid((y-0.09) / 0.9)
            self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._knn_f = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_abs = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_dists = None
        self.knn_idx = None

        self._visibility_tracing = None

        self.use_pbr = True

        self.setup_functions()
        self.use_app = False

        

        if self.use_pbr:
            self._normal = torch.empty(0)
            self._base_color = torch.empty(0)
            self._roughness = torch.empty(0)
            self._incidents_dc = torch.empty(0)
            self._incidents_rest = torch.empty(0)
            self._scatters_dc = torch.empty(0)         # for the scattersing light in the object
            self._scatters_rest = torch.empty(0)
            self._visibility_dc = torch.empty(0)
            self._visibility_rest = torch.empty(0)
        self.base_color_scale = torch.ones(3, dtype=torch.float, device="cuda")

    def capture(self):
        captured = [
            self.active_sh_degree,
            self._xyz,
            self._knn_f,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.denom_abs,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        ]
        if self.use_pbr:
            captured.extend([
                self._normal,
                self._base_color,
                self._roughness,
                self._incidents_dc,
                self._incidents_rest,
                self._scatters_dc,
                self._scatters_rest,
                self._visibility_dc,
                self._visibility_rest,
            ])
        return captured
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._knn_f,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.max_weight,
        xyz_gradient_accum, 
        xyz_gradient_accum_abs,
        denom,
        denom_abs,
        opt_dict, 
        self.spatial_lr_scale,
        ) = model_args[:16]
        if len(model_args) > 16 and self.use_pbr:
            (self._normal,
             self._base_color,
             self._roughness,
             self._incidents_dc,
             self._incidents_rest,
             self._scatters_dc,
             self._scatters_rest,
             self._visibility_dc,
             self._visibility_rest) = model_args[16:]
        
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.denom_abs = denom_abs
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
        
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_incidents(self):
        """SH"""
        incidents_dc = self._incidents_dc
        incidents_rest = self._incidents_rest
        return torch.cat((incidents_dc, incidents_rest), dim=1)

    @property
    def get_scattersColor(self):
        scatters_dc = self._scatters_dc
        scatters_rest = self._scatters_rest
        return  torch.cat((scatters_dc, scatters_rest), dim=1)
    
    @property
    def get_visibility(self):
        """SH"""
        visibility_dc = self._visibility_dc
        visibility_rest = self._visibility_rest
        return torch.cat((visibility_dc, visibility_rest), dim=1)
    
    @property
    def get_base_color(self):
        return self.base_color_activation(self._base_color) * self.base_color_scale[None, :]

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)

    @property
    def get_brdf(self):
        return torch.cat([self.get_base_color, self.get_roughness], dim=-1)
    
    def get_normal_p(self, view_cam=None):
        normal_global = self.normal_activation(self._normal)
        if view_cam == None:
            return normal_global
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    

    def get_normal(self, view_cam=None):
        normal_global = self.get_smallest_axis()
        if view_cam == None:
            return normal_global
        
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling,
                                          1 / scaling_modifier,
                                          self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # def getPointProbabilities(self, points):
    #     '''
    #     points: input sample points [N, 3]
        
    #     return: [N, ]
    #     '''
    #     centers = self.get_xyz # [M, 3]
    #     dists, idxs, _ = knn_points(points.unsqueeze(0), centers.unsqueeze(0), K=100)
    #     knn_centers = centers[idxs] # [N, K, 3]
    #     knn_inv_cov = (self.get_inverse_covariance())[idxs] # [N, K, 3, 3]
    #     knn_scales = (self.get_scaling)[idxs] # [N, K, 3]
    #     knn_opacity = (self.get_opacity)[idxs] # [N, K, ]

    #     diff_vec = diff_vec = points[:, None, :] - knn_centers # [N, K, 3]
    #     temp = torch.matmul(diff_vec, knn_inv_cov)
    #     mahalanobis_dist = torch.einsum('ijk, ijk -> ij', temp, diff_vec)  # [N, K]
    #     normalizing_const = (2 * torch.pi) ** (3 / 2) * torch.sqrt(torch.det(knn_scales**2))
    #     probs = knn_opacity * torch.exp(-0.5 * mahalanobis_dist) / normalizing_const  # [N, K]

    #     probs = probs.sum(dim=1)
    #     return probs
        
    @torch.no_grad()
    def update_visibility(self, sample_num):
        raytracer = RayTracer(self.get_xyz, self.get_scaling, self.get_rotation)
        gaussians_xyz = self.get_xyz
        gaussians_inverse_covariance = self.get_inverse_covariance()
        gaussians_opacity = self.get_opacity[:, 0]
        gaussians_normal = self.get_normal_p()
        incident_visibility_results = []
        incident_dirs_results = []
        incident_areas_results = []
        chunk_size = gaussians_xyz.shape[0] // ((sample_num - 1) // 24 + 1)
        for offset in range(0, gaussians_xyz.shape[0], chunk_size):
            incident_dirs, incident_areas = sample_incident_rays(gaussians_normal[offset:offset + chunk_size], False,
                                                    sample_num)
            # trace_results = raytracer.trace_visibility(
            #     gaussians_xyz[offset:offset + chunk_size, None].expand_as(incident_dirs),
            #     incident_dirs,
            #     gaussians_xyz,
            #     gaussians_inverse_covariance,
            #     gaussians_opacity,
            #     gaussians_normal)
            # incident_visibility = trace_results["visibility"]
            # incident_visibility_results.append(incident_visibility)
            incident_dirs_results.append(incident_dirs)
            incident_areas_results.append(incident_areas)
        # incident_visibility_result = torch.cat(incident_visibility_results, dim=0)
        incident_dirs_result = torch.cat(incident_dirs_results, dim=0)
        incident_areas_result = torch.cat(incident_areas_results, dim=0)
        # self._visibility_tracing = incident_visibility_result
        self._incident_dirs = incident_dirs_result
        self._incident_areas = incident_areas_result

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001))
        # print(f"new scale {torch.quantile(dist, 0.1)}")
        scales = torch.log(dist)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        knn_f = torch.randn((fused_point_cloud.shape[0], 6)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_pbr:
            self._normal = nn.Parameter(fused_normal.requires_grad_(True))

            base_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            roughness = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

            self._base_color = nn.Parameter(base_color.requires_grad_(True))
            self._roughness = nn.Parameter(roughness.requires_grad_(True))

            incidents = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            self._incidents_dc = nn.Parameter(incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._incidents_rest = nn.Parameter(incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

            scatters = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            self._scatters_dc = nn.Parameter(scatters[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._scatters_rest = nn.Parameter(scatters[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

            visibility = torch.zeros((self._xyz.shape[0], 1, 4 ** 2)).float().cuda()
            self._visibility_dc = nn.Parameter(visibility[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._visibility_rest = nn.Parameter(visibility[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.abs_split_radii2D_threshold = training_args.abs_split_radii2D_threshold
        self.max_abs_split_points = training_args.max_abs_split_points
        self.max_all_points = training_args.max_all_points
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._knn_f], 'lr': 0.01, "name": "knn_f"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        
        if self.use_pbr:
            if training_args.light_rest_lr < 0:
                training_args.light_rest_lr = training_args.light_lr / 20.0
            if training_args.visibility_rest_lr < 0:
                training_args.visibility_rest_lr = training_args.visibility_lr / 20.0

            l.extend([
                {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
                {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._incidents_dc], 'lr': training_args.light_lr, "name": "incidents_dc"},
                {'params': [self._incidents_rest], 'lr': training_args.light_rest_lr, "name": "incidents_rest"},
                {'params': [self._scatters_dc], 'lr': training_args.light_lr, 'name': 'scatters_dc'},
                {'params': [self._scatters_rest], 'lr':  training_args.light_rest_lr, 'name': 'scatters_rest'},
                {'params': [self._visibility_dc], 'lr': training_args.visibility_lr, "name": "visibility_dc"},
                {'params': [self._visibility_rest], 'lr': training_args.visibility_rest_lr, "name": "visibility_rest"},
            ])
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def clip_grad(self, norm=1.0):
        for group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"][0], norm)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        
        if self.use_pbr:
            for i in range(self._base_color.shape[1]):
                l.append('base_color_{}'.format(i))
            l.append('roughness')
            for i in range(self._incidents_dc.shape[1] * self._incidents_dc.shape[2]):
                l.append('incidents_dc_{}'.format(i))
            for i in range(self._incidents_rest.shape[1] * self._incidents_rest.shape[2]):
                l.append('incidents_rest_{}'.format(i))
            for i in range(self._scatters_dc.shape[1] * self._scatters_dc.shape[2]):
                l.append('scatters_dc_{}'.format(i))
            for i in range(self._scatters_rest.shape[1] * self._scatters_rest.shape[2]):
                l.append('scatters_rest_{}'.format(i))
            for i in range(self._visibility_dc.shape[1] * self._visibility_dc.shape[2]):
                l.append('visibility_dc_{}'.format(i))
            for i in range(self._visibility_rest.shape[1] * self._visibility_rest.shape[2]):
                l.append('visibility_rest_{}'.format(i))
        return l

    def save_ply(self, path, mask=None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attributes_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

        if self.use_pbr:
            attributes_list.extend([
                self._base_color.detach().cpu().numpy(),
                self._roughness.detach().cpu().numpy(),
                self._incidents_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._incidents_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._scatters_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._scatters_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._visibility_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._visibility_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
            ])

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                           np.asarray(plydata.elements[0]["ny"]),
                           np.asarray(plydata.elements[0]["nz"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        if self.use_pbr:
            self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
            base_color_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("base_color")]
            base_color_names = sorted(base_color_names, key=lambda x: int(x.split('_')[-1]))
            base_color = np.zeros((xyz.shape[0], len(base_color_names)))
            for idx, attr_name in enumerate(base_color_names):
                base_color[:, idx] = np.asarray(plydata.elements[0][attr_name])

            roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

            self._base_color = nn.Parameter(
                torch.tensor(base_color, dtype=torch.float, device="cuda").requires_grad_(True))
            self._roughness = nn.Parameter(
                torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))

            incidents_dc = np.zeros((xyz.shape[0], 3, 1))
            incidents_dc[:, 0, 0] = np.asarray(plydata.elements[0]["incidents_dc_0"])
            incidents_dc[:, 1, 0] = np.asarray(plydata.elements[0]["incidents_dc_1"])
            incidents_dc[:, 2, 0] = np.asarray(plydata.elements[0]["incidents_dc_2"])
            extra_incidents_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("incidents_rest_")]
            extra_incidents_names = sorted(extra_incidents_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_incidents_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            incidents_extra = np.zeros((xyz.shape[0], len(extra_incidents_names)))
            for idx, attr_name in enumerate(extra_incidents_names):
                incidents_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            incidents_extra = incidents_extra.reshape((incidents_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            self._incidents_dc = nn.Parameter(torch.tensor(incidents_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._incidents_rest = nn.Parameter(torch.tensor(incidents_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

            scatters_dc = np.zeros([xyz.shape[0], 3, 1])
            scatters_dc[:, 0, 0] = np.asarray(plydata.elements[0]['scatters_dc_0'])
            scatters_dc[:, 1, 0] = np.asarray(plydata.elements[0]['scatters_dc_1'])
            scatters_dc[:, 2, 0] = np.asarray(plydata.elements[0]['scatters_dc_2'])
            extra_scatters_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scatters_rest_")]
            extra_scatters_names = sorted(extra_scatters_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_scatters_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            scatters_extra = np.zeros([xyz.shape[0], len(extra_scatters_names)])
            for idx, attr_name in enumerate(extra_scatters_names):
                scatters_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            scatters_extra = scatters_extra.reshape([scatters_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1])
            self._scatters_dc = nn.Parameter(torch.tensor(scatters_dc, dtype=torch.float, device='cuda').transpose(1,2).contiguous().requires_grad_(True))
            self._scatters_rest = nn.Parameter(torch.tensor(scatters_extra, dtype=torch.float, device='cuda').transpose(1,2).contiguous().requires_grad_(True))

            
            visibility_dc = np.zeros((xyz.shape[0], 1, 1))
            visibility_dc[:, 0, 0] = np.asarray(plydata.elements[0]["visibility_dc_0"])
            extra_visibility_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("visibility_rest_")]
            extra_visibility_names = sorted(extra_visibility_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_visibility_names) == 4 ** 2 - 1
            visibility_extra = np.zeros((xyz.shape[0], len(extra_visibility_names)))
            for idx, attr_name in enumerate(extra_visibility_names):
                visibility_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            visibility_extra = visibility_extra.reshape((visibility_extra.shape[0], 1, 4 ** 2 - 1))
            self._visibility_dc = nn.Parameter(torch.tensor(visibility_dc, dtype=torch.float, device="cuda").transpose(
                1, 2).contiguous().requires_grad_(True))
            self._visibility_rest = nn.Parameter(
                torch.tensor(visibility_extra, dtype=torch.float, device="cuda").transpose(
                    1, 2).contiguous().requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.denom_abs = self.denom_abs[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]

        if self.use_pbr:
            self._normal = optimizable_tensors["normal"]
            self._base_color = optimizable_tensors["base_color"]
            self._roughness = optimizable_tensors["roughness"]
            self._incidents_dc = optimizable_tensors["incidents_dc"]
            self._incidents_rest = optimizable_tensors["incidents_rest"]
            self._scatters_dc = optimizable_tensors["scatters_dc"]
            self._scatters_rest = optimizable_tensors["scatters_rest"]
            self._visibility_dc = optimizable_tensors["visibility_dc"]
            self._visibility_rest = optimizable_tensors["visibility_rest"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_tensors_dict, reset_param = True):
        # d = {"xyz": new_xyz,
        # "knn_f": new_knn_f,
        # "f_dc": new_features_dc,
        # "f_rest": new_features_rest,
        # "opacity": new_opacities,
        # "scaling" : new_scaling,
        # "rotation" : new_rotation}

        # if self.use_pbr:
        #     d.update({
        #         "normal": new_normal,
        #         "base_color": new_base_color,
        #         "roughness": new_roughness,
        #         "incidents_dc": new_incidents_dc,
        #         "incidents_rest": new_incidents_rest,
        #         "scatters_dc": new_scatters_dc,
        #         "scatters_rest": new_scatters_rest,
        #         "visibility_dc": new_visibility_dc,
        #         "visibility_rest": new_visibility_rest,
        #     })

        optimizable_tensors = self.cat_tensors_to_optimizer(new_tensors_dict)
        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.use_pbr:
            self._normal = optimizable_tensors["normal"]
            self._base_color = optimizable_tensors["base_color"]
            self._roughness = optimizable_tensors["roughness"]
            self._incidents_dc = optimizable_tensors["incidents_dc"]
            self._incidents_rest = optimizable_tensors["incidents_rest"]
            self._scatters_dc = optimizable_tensors["scatters_dc"]
            self._scatters_rest = optimizable_tensors["scatters_rest"]
            self._visibility_dc = optimizable_tensors["visibility_dc"]
            self._visibility_rest = optimizable_tensors["visibility_rest"]
        
        if reset_param:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, max_radii2D, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grads_abs = torch.zeros((n_init_points), device="cuda")
        padded_grads_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points), device="cuda")
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            padded_grad[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(padded_grad, (1.0-ratio))
            selected_pts_mask = torch.where(padded_grad > threshold, True, False)
            # print(f"split {selected_pts_mask.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")
        else:
            padded_grads_abs[selected_pts_mask] = 0
            mask = (torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) & (padded_max_radii2D > self.abs_split_radii2D_threshold)
            padded_grads_abs[~mask] = 0
            selected_pts_mask_abs = torch.where(padded_grads_abs >= grad_abs_threshold, True, False)
            limited_num = min(self.max_all_points - n_init_points - selected_pts_mask.sum(), self.max_abs_split_points)
            if selected_pts_mask_abs.sum() > limited_num:
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(padded_grads_abs, (1.0-ratio))
                selected_pts_mask_abs = torch.where(padded_grads_abs > threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
            # print(f"split {selected_pts_mask.sum()}, abs {selected_pts_mask_abs.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        # new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        # new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        # new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # new_knn_f = self._knn_f[selected_pts_mask].repeat(N,1)

        new_tensors_dict = {
            "xyz": torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1),
            "knn_f": self._knn_f[selected_pts_mask].repeat(N,1),
            "f_dc": self._features_dc[selected_pts_mask].repeat(N,1,1),
            "f_rest": self._features_rest[selected_pts_mask].repeat(N,1,1),
            "opacity": self._opacity[selected_pts_mask].repeat(N,1),
            "scaling" : self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)),
            "rotation" : self._rotation[selected_pts_mask].repeat(N,1)
        }
        if self.use_pbr:
            new_tensors_dict.update({
                "normal": self._normal[selected_pts_mask].repeat(N, 1),
                "base_color": self._base_color[selected_pts_mask].repeat(N, 1),
                "roughness": self._roughness[selected_pts_mask].repeat(N, 1),
                "incidents_dc": self._incidents_dc[selected_pts_mask].repeat(N, 1, 1),
                "incidents_rest": self._incidents_rest[selected_pts_mask].repeat(N, 1, 1),
                "scatters_dc": self._scatters_dc[selected_pts_mask].repeat(N, 1, 1),
                "scatters_rest": self._scatters_rest[selected_pts_mask].repeat(N, 1, 1),
                "visibility_dc": self._visibility_dc[selected_pts_mask].repeat(N, 1, 1),
                "visibility_rest": self._visibility_rest[selected_pts_mask].repeat(N, 1, 1)
            })
        self.densification_postfix(new_tensors_dict)

        # args = [new_xyz, new_normal, new_knn_f, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation]
        # if self.use_pbr:
        #     new_base_color = self._base_color[selected_pts_mask].repeat(N, 1)
        #     new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
        #     new_incidents_dc = self._incidents_dc[selected_pts_mask].repeat(N, 1, 1)
        #     new_incidents_rest = self._incidents_rest[selected_pts_mask].repeat(N, 1, 1)
        #     new_scatters_dc = self._scatters_dc[selected_pts_mask].repeat(N, 1, 1)
        #     new_scatters_rest = self._scatters_rest[selected_pts_mask].repeat(N, 1, 1)
        #     new_visibility_dc = self._visibility_dc[selected_pts_mask].repeat(N, 1, 1)
        #     new_visibility_rest = self._visibility_rest[selected_pts_mask].repeat(N, 1, 1)
        #     args.extend([
        #         new_base_color,
        #         new_roughness,
        #         new_incidents_dc,
        #         new_incidents_rest,
        #         new_scatters_dc,
        #         new_scatters_rest,
        #         new_visibility_dc,
        #         new_visibility_rest,
        #     ])
        # self.densification_postfix(*args)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            grads_tmp = grads.squeeze().clone()
            grads_tmp[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(grads_tmp, (1.0-ratio))
            selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

        if selected_pts_mask.sum() > 0:
            # print(f"clone {selected_pts_mask.sum()}")
            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])

            # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            # new_normal = self._normal[selected_pts_mask]
            # new_features_dc = self._features_dc[selected_pts_mask]
            # new_features_rest = self._features_rest[selected_pts_mask]
            # new_opacities = self._opacity[selected_pts_mask]
            # new_scaling = self._scaling[selected_pts_mask]
            # new_rotation = self._rotation[selected_pts_mask]
            # new_knn_f = self._knn_f[selected_pts_mask]

            # args = [new_xyz, new_normal, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation]
            # if self.use_pbr:
            #     new_base_color = self._base_color[selected_pts_mask]
            #     new_roughness = self._roughness[selected_pts_mask]
            #     new_incidents_dc = self._incidents_dc[selected_pts_mask]
            #     new_incidents_rest = self._incidents_rest[selected_pts_mask]
            #     new_scatters_dc = self._scatters_dc[selected_pts_mask]
            #     new_scatters_rest = self._scatters_rest[selected_pts_mask]
            #     new_visibility_dc = self._visibility_dc[selected_pts_mask]
            #     new_visibility_rest = self._visibility_rest[selected_pts_mask]

            #     args.extend([
            #         new_base_color,
            #         new_roughness,
            #         new_incidents_dc,
            #         new_incidents_rest,
            #         new_scatters_dc,
            #         new_scatters_rest,
            #         new_visibility_dc,
            #         new_visibility_rest,
            #     ])
            # self.densification_postfix(*args)
        
            new_tensors_dict = {
                "xyz": torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask],
                "knn_f": self._knn_f[selected_pts_mask],
                "f_dc": self._features_dc[selected_pts_mask],
                "f_rest": self._features_rest[selected_pts_mask],
                "opacity": self._opacity[selected_pts_mask],
                "scaling" : self._scaling[selected_pts_mask],
                "rotation" : self._rotation[selected_pts_mask]
            }
            if self.use_pbr:
                new_tensors_dict.update({
                    "normal": self._normal[selected_pts_mask],
                    "base_color": self._base_color[selected_pts_mask],
                    "roughness": self._roughness[selected_pts_mask],
                    "incidents_dc": self._incidents_dc[selected_pts_mask],
                    "incidents_rest": self._incidents_rest[selected_pts_mask],
                    "scatters_dc": self._scatters_dc[selected_pts_mask],
                    "scatters_rest": self._scatters_rest[selected_pts_mask],
                    "visibility_dc": self._visibility_dc[selected_pts_mask],
                    "visibility_rest": self._visibility_rest[selected_pts_mask]
                })
            self.densification_postfix(new_tensors_dict)

    def densify_and_prune(self, max_grad, abs_max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads_abs = self.xyz_gradient_accum_abs / self.denom_abs
        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0
        max_radii2D = self.max_radii2D.clone()

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, grads_abs, abs_max_grad, extent, max_radii2D)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        # print(f"all points {self._xyz.shape[0]}")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.denom_abs[update_filter] += 1

    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale/2)-1,0)
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                         points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask
    
    def get_points_from_depth(self, fov_camera, depth, scale=1):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2)
        return pts
    
    # <start> new add mcmc 03-14

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "knn_f": self._knn_f,            
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}

        if self.use_pbr:
            tensors_dict.update({
                "normal":         self._normal,
                "base_color":     self._base_color,
                "roughness":      self._roughness,
                "incidents_dc":   self._incidents_dc,
                "incidents_rest": self._incidents_rest,
                "scatters_dc":    self._scatters_dc,
                "scatters_rest":  self._scatters_rest,
                "visibility_dc":  self._visibility_dc,
                "visibility_rest": self._visibility_rest,
            })

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if inds is not None:
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.use_pbr:
            self._normal = optimizable_tensors["normal"]
            self._base_color = optimizable_tensors["base_color"]
            self._roughness = optimizable_tensors["roughness"]
            self._incidents_dc = optimizable_tensors["incidents_dc"]
            self._incidents_rest = optimizable_tensors["incidents_rest"]
            self._scatters_dc = optimizable_tensors["scatters_dc"]
            self._scatters_rest = optimizable_tensors["scatters_rest"]
            self._visibility_dc = optimizable_tensors["visibility_dc"]
            self._visibility_rest = optimizable_tensors["visibility_rest"]

        torch.cuda.empty_cache()
        
        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    
    def relocate_gs(self, dead_mask=None):

        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        self._knn_f[dead_indices] = self._knn_f[reinit_idx]
        if self.use_pbr:
            self._normal[dead_indices] = self._normal[reinit_idx]
            self._base_color[dead_indices] = self._base_color[reinit_idx]
            self._roughness[dead_indices] = self._roughness[reinit_idx]
            self._incidents_dc[dead_indices] = self._incidents_dc[reinit_idx]
            self._incidents_rest[dead_indices] = self._incidents_rest[reinit_idx]
            self._scatters_dc[dead_indices] = self._scatters_dc[reinit_idx]
            self._scatters_rest[dead_indices] = self._scatters_rest[reinit_idx]
            self._visibility_dc[dead_indices] = self._visibility_dc[reinit_idx]
            self._visibility_rest[dead_indices] = self._visibility_rest[reinit_idx]
        
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx) 
        

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1) 
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)
        new_knn_f = self._knn_f[add_idx]
        new_tensors_dict = {
            "xyz": new_xyz,
            "knn_f": new_knn_f,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling" : new_scaling,
            "rotation" : new_rotation
        }
        if self.use_pbr:
            new_tensors_dict.update({
                "normal": self._normal[add_idx],
                "base_color": self._base_color[add_idx],
                "roughness": self._roughness[add_idx],
                "incidents_dc": self._incidents_dc[add_idx],
                "incidents_rest": self._incidents_rest[add_idx],
                "scatters_dc": self._scatters_dc[add_idx],
                "scatters_rest": self._scatters_rest[add_idx],
                "visibility_dc": self._visibility_dc[add_idx],
                "visibility_rest": self._visibility_rest[add_idx]
            })

        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        self.densification_postfix(new_tensors_dict, reset_param=False)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    def add_noise(self, xyz_lr):
        with torch.no_grad():
            L = build_scaling_rotation(self.get_scaling, self.get_rotation)
            actual_covariance = L @ L.transpose(1, 2)

            def op_sigmoid(x, k=100, x0=0.995):
                return 1 / (1 + torch.exp(-k * (x - x0)))
            
            noise = torch.randn_like(self._xyz) * (op_sigmoid(1 - self.get_opacity)) * 5e5 * xyz_lr
            noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
            self._xyz.add_(noise)