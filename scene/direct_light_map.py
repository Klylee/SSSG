import torch
import torch.nn as nn
import numpy as np
from arguments import OptimizationParams
from utils.sh_utils import eval_sh, eval_sh_coef
import nvdiffrast.torch as dr
import torch.nn.functional as F

class DirectLightMap:

    def __init__(self, H=128, light_init=0.5):
        self.H = H
        self.W = H * 2
        env = (light_init * torch.rand((1, self.H, self.W, 3))).float().cuda().requires_grad_(True)
        self.env = nn.Parameter(env)
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.light_strength = nn.Parameter(torch.tensor([5.0], dtype=torch.float32).cuda().requires_grad_(True))
        
    def training_setup(self, training_args: OptimizationParams):
        l = [
            {'params': [self.env], 'lr': training_args.env_lr, "name": "env"},
            {'params': [self.light_strength], 'lr': training_args.env_lr, 'name': 'light'}
        ]

        self.optimizer = torch.optim.Adam(l, eps=1e-15)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def capture(self):
        captured_list = [
            self.env,
            self.light_strength,
            self.optimizer.state_dict(),
        ]

        return captured_list

    def restore(self, checkpoint_path, training_arg):
        pass

    def create_from_ckpt(self, checkpoint_path, training_arg=None, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.env, self.light_strength, opt_dict) = model_args[:3]
        if training_arg != None:
            self.training_setup(training_arg)

        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter
    
    def direct_light(self, dirs, transform=None):
        shape = dirs.shape
        dirs = dirs.reshape(-1, 3)

        envir_map = self.get_env.permute(0, 3, 1, 2) # [1, 3, H, W]
        phi = torch.arccos(dirs[:, 2]).reshape(-1) - 1e-6
        theta = torch.atan2(dirs[:, 1], dirs[:, 0]).reshape(-1)
        # normalize to [-1, 1]
        query_y = (phi / np.pi) * 2 - 1
        query_x = - theta / np.pi
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)
        light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
    
        return light_rgbs.reshape(*shape)

    def upsample(self):
        self.env = nn.Parameter(F.interpolate(self.env.data.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1).requires_grad_(True))
        self.H *= 2
        self.W *= 2
        
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = F.interpolate(stored_state["exp_avg"].permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
                stored_state["exp_avg_sq"] = F.interpolate(stored_state["exp_avg_sq"].permute(0, 3, 1, 2), scale_factor=2, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)

                del self.optimizer.state[group['params'][0]]

                group["params"][0] = nn.Parameter(self.env)
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(self.env)

    @property
    def get_env(self):
        return F.softplus(self.env)
    
    @property
    def colocated_light(self):
        return F.softplus(self.light_strength)