from typing import Literal
from networkx import local_constraint
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from pytorch_msssim import MS_SSIM
from models.filters import BilateralFilter, Gradient
from models.models2 import FuseInput, FuseOutput




class ALL_Loss(nn.Module):
    def __init__(self, a: float, b: float, c: float, d: float):
        super().__init__()

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.bila = BilateralFilter()
        self.grad = Gradient()
        self.ssim = MS_SSIM(data_range=1.0, channel=1)
        # self.downsample = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)

    def intensity_loss(self, vi: Tensor, ir: Tensor, fu: Tensor):
        if len(fu.shape) == 3:
            fu = fu.unsqueeze(1)
        return F.l1_loss(fu, torch.max(vi, ir))

    def grad_loss(self, vi: Tensor, ir: Tensor, fu: Tensor):
        if len(fu.shape) == 3:
            fu = fu.unsqueeze(1)
        return F.l1_loss(self.grad(fu), torch.max(self.grad(self.bila(ir)), self.grad(vi)))

    def ssim_loss(self, vi: Tensor, ir: Tensor, fu: Tensor) -> Tensor:
        if len(fu.shape) == 3:
            fu = fu.unsqueeze(1)
        return 1 - 0.5 * (self.ssim(fu, vi) + self.ssim(fu, ir))

    def local_contrast(self, image: Tensor, kernel_size: int = 3) -> Tensor:
        mean_filter = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        if image.is_cuda:
            mean_filter = mean_filter.cuda()
        local_mean = F.conv2d(image, mean_filter, padding=kernel_size // 2)
        local_contrast = torch.abs(image - local_mean)
        return local_contrast

    def fu_loss(self, vi: Tensor, ir: Tensor, fu: Tensor) -> Tensor:
        if len(fu.shape) == 3:
            fu = fu.unsqueeze(1)
      
        l_pixel = F.l1_loss(fu, (vi + ir ) /2)
        l_intensity = self.intensity_loss(vi, ir, fu)
        l_ssim = self.ssim_loss(vi, ir, fu)
        l_grad = self.grad_loss(vi, ir, fu)
        return self.a * l_pixel + self.b * l_intensity + self.c * l_grad + self.d * l_ssim
        
    


    def forward(self, inputs: FuseInput, outputs: FuseOutput) -> Tensor:
        
        vi = inputs.vi / 2 + 0.5
        ir = inputs.ir / 2 + 0.5

        fu = outputs.fusion / 2 + 0.5




        loss_fuse = self.fu_loss(vi, ir, fu)
        return loss_fuse

