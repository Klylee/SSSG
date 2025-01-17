import torch
import numpy as np


def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1) # (H, W, 3)

    return directions
