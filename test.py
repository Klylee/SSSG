from torchvision import utils
import torch

depth = torch.randn(1, 100, 100)
depth = depth.detach().cpu()
depth = -depth.clip(0, 3)
depth_i = (depth - depth.min()) / ( depth.max()- depth.min() + 1e-20)
utils.save_image(depth_i, 'depth.png', nrow=1, normalize=True)