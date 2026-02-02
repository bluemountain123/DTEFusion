import torch
from torch import Tensor, nn 
import torch.nn.functional as F

from models.base import DWConv2d


def gaussian_kernel(kernel_size: int = 5, sigma: float | None = None):
    sigma = 0.3 * ((kernel_size - 1) / 2.0 - 1) + 0.8 if sigma is None else sigma
    x_grid = torch.arange(kernel_size).repeat(kernel_size, 1)
    xy_grid = torch.stack([x_grid, x_grid.T], dim=-1).float()

    center = kernel_size >> 1
    weight = torch.exp(-torch.sum((xy_grid - center) ** 2., dim=-1) / (2 * sigma ** 2))
    # weight /= torch.sum(weight)
    return weight / torch.sum(weight)

class Gradient(nn.Module):
    def __init__(
        self,
        channels: int = 1
    ):
        super().__init__()
        sobel_x = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ])
        self.sobel_x = DWConv2d(channels, sobel_x, with_bias=False)
        self.sobel_y = DWConv2d(channels, sobel_x.T, with_bias=False)

    def forward(self, x: Tensor):
        dx: Tensor = self.sobel_x(x)
        dy: Tensor = self.sobel_y(x)
        return dx.abs() + dy.abs()

class GaussianFilter(nn.Module):
    def __init__(
        self,
        kernel_size: int = 5,
        sigma: float | None = None,
        channels: int = 1
    ):
        super().__init__()
        self.gaussian_weight = gaussian_kernel(kernel_size, sigma)
        weight = self.gaussian_weight.repeat(channels, 1, 1, 1)
        pad = (kernel_size - 1) >> 1
        self.filter = nn.Conv2d(
            channels, channels, kernel_size, bias=False,
            padding=pad, padding_mode='replicate', # reflect
            groups=channels
        )
        self.filter.weight = nn.Parameter(weight, requires_grad=False)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.filter(x)

class BilateralFilter(nn.Module):
    def __init__(
        self,
        kernel_size: int = 5,
        sigma_space: float | None = None,
        sigma_density: float | None = None
    ):
        super().__init__()
        self.sigma_space = 0.3 * ((kernel_size-1) * 0.5 - 1) + 0.8 if sigma_space is None else sigma_space
        self.sigma_density = self.sigma_space if sigma_density is None else sigma_density

        self.pad = (kernel_size - 1) >> 1
        self.kernel_size = kernel_size
        
        self.gaussian_weights = nn.Parameter(gaussian_kernel(kernel_size, self.sigma_space), requires_grad=False)

    def forward(self, x: Tensor):
        x_pad = F.pad(x, pad=[self.pad, self.pad, self.pad, self.pad], mode='replicate')
        x_patches = x_pad.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        
        patch_dim = x_patches.dim()

        diff_density = x_patches - x.unsqueeze(-1).unsqueeze(-1)
        weight_density = torch.exp(-(diff_density ** 2) / (2 * self.sigma_density ** 2))

        weight_density /= weight_density.sum(dim=(-1, -2), keepdim=True)

        weight_density_shape = (1, ) * (patch_dim - 2) + (self.kernel_size, self.kernel_size)
        weight_space = self.gaussian_weights.view(*weight_density_shape).expand_as(weight_density)

        weight = weight_density * weight_space

        return (weight * x_patches).sum(dim=(-1, -2)) / weight.sum(dim=(-1, -2))
