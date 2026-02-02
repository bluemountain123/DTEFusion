# metric.py

import numpy as np
from scipy.stats import entropy
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur
from pytorch_msssim import ssim, ms_ssim

def diff(img: Tensor):
    dx = img.diff(dim=1)[:, :, :-1]
    dy = img.diff(dim=2)[:, :-1, :]
    return dx, dy

def entropy_calculation(x: Tensor):
    """
    Calculate the entropy of a given probability distribution.

    Args:
        x (torch.Tensor): 1D probability distribution tensor
    Returns:
        float: Entropy value
    """
    en = - (x * torch.log2(x).nan_to_num()).sum().item()
    return en

def cross_entropy(x: Tensor, y: Tensor):
    ce = (x * torch.log2(x / y).nan_to_num(posinf=0.0, neginf=0.0)).sum().item()
    return ce

def mutual_information(x: Tensor, y: Tensor):
    hist = histogram2d(x, y)
    mi = entropy_calculation(hist.sum(dim=0)) + entropy_calculation(hist.sum(dim=1)) - entropy_calculation(hist)
    return mi

def correlation_coefficient(x: Tensor, y: Tensor):
    centralized_x = x - x.mean()
    centralized_y = y - y.mean()
    nume = (centralized_x * centralized_y).sum().item()
    deno = torch.sqrt(centralized_x.square().sum() * centralized_y.square().sum()).item()
    return nume / deno if deno != 0 else 0.0

def mean_squared_error(src: Tensor, dst: Tensor, w=0.5):
    return (w * F.mse_loss(src, dst) + (1 - w) * F.mse_loss(dst, src)).item()

def calc_SD(fu: Tensor):
    SD = fu.std().item()
    return SD

def calc_SF(fu: Tensor):
    dx, dy = diff(fu)
    rf = dx.square().sum() / fu.numel()
    cf = dy.square().sum() / fu.numel()
    SF = (rf + cf).sqrt().item()
    return SF

def calc_AG(fu: Tensor):
    C, H, W = fu.shape
    dx, dy = torch.gradient(fu, dim=[1, 2])
    AG = ((dx.square() + dy.square()) / 2).sqrt().sum() / ((H - 1) * (W - 1))
    return AG.item()

def calc_CC(vi: Tensor, ir: Tensor, fu: Tensor, w=0.5):
    CC = w * correlation_coefficient(vi, fu) + (1 - w) * correlation_coefficient(ir, fu)
    return CC

def calc_SCD(vi: Tensor, ir: Tensor, fu: Tensor):
    vif = correlation_coefficient(fu - ir, vi)
    irf = correlation_coefficient(fu - vi, ir)
    SCD = vif + irf
    return SCD

def histogram2d(src: Tensor, dst: Tensor):
    """
    Calculate 2D histogram from PyTorch tensors.

    Args:
        src (torch.Tensor): Source tensor, assumed range is [0, 1].
        dst (torch.Tensor): Destination tensor, assumed range is [0, 1].

    Returns:
        torch.Tensor: 2D histogram
    """
    src_np = src.cpu().numpy().flatten()
    dst_np = dst.cpu().numpy().flatten()
    hist, _, _ = np.histogram2d(src_np, dst_np, bins=256, range=[[0, 1], [0, 1]], density=True)
    return torch.tensor(hist, device=src.device)

def calc_EN(fu: Tensor):
    """
    Calculate the entropy of the fused image.

    Args:
        fu (torch.Tensor): Fused image tensor, assumed range is [0, 1].

    Returns:
        float: Calculated entropy value.
    """
    # Move tensor to CPU and convert to NumPy array
    h_f = fu.cpu().numpy().flatten()

    # Calculate histogram, bins=256 for grayscale image, range is [0, 1]
    hist, _ = np.histogram(h_f, bins=256, range=(0, 1), density=True)

    # Avoid log(0) issues by adding a small value
    hist += 1e-10

    # Calculate entropy
    EN = entropy(hist)
    
    return EN

def calc_CE(vi: Tensor, ir: Tensor, fu: Tensor, w=0.5):
    h_v = histogram(vi)
    h_i = histogram(ir)
    h_f = histogram(fu)
    CE = w * cross_entropy(torch.tensor(h_v), torch.tensor(h_f)) + (1 - w) * cross_entropy(torch.tensor(h_i), torch.tensor(h_f))
    return CE

def calc_MI(vi: Tensor, ir: Tensor, fu: Tensor):
    MI = mutual_information(vi, fu) + mutual_information(ir, fu)
    return MI

def calc_PSNR(vi: Tensor, ir: Tensor, fu: Tensor):
    MSE = calc_MSE(vi, ir, fu)
    PSNR = 20 * (1. / np.sqrt(MSE)).log10()
    return PSNR

def vifp_mscale1(src: Tensor, dst: Tensor) -> Tensor:
    x, y = src, dst

    sigma_nsq = 2
    nume, deno = torch.tensor(0, dtype=torch.float64, device=src.device), torch.tensor(0, dtype=torch.float64, device=src.device)
    for scale in range(4):
        kernel_size = (1 << (4 - scale)) | 1
        sigma = kernel_size / 5.0
        mu_x = gaussian_blur(x, kernel_size=kernel_size, sigma=sigma) # type: ignore
        mu_y = gaussian_blur(y, kernel_size=kernel_size, sigma=sigma) # type: ignore
        mu_x_square = mu_x.square()
        mu_y_square = mu_y.square()
        mu_xy_square = mu_x * mu_y

        sigma_x_square = gaussian_blur(x.square(), kernel_size=kernel_size, sigma=sigma) - mu_x_square # type: ignore
        sigma_y_square = gaussian_blur(y.square(), kernel_size=kernel_size, sigma=sigma) - mu_y_square # type: ignore
        sigma_xy_square = gaussian_blur(x * y, kernel_size=kernel_size, sigma=sigma) - mu_xy_square # type: ignore

        sigma_x_square = sigma_x_square.clamp(min=0)
        sigma_y_square = sigma_y_square.clamp(min=0)
        g = torch.where(sigma_xy_square > 0,
            (sigma_xy_square / sigma_x_square).nan_to_num(torch.inf),
            torch.where(sigma_xy_square < 0,
                (sigma_xy_square / sigma_x_square).nan_to_num(-torch.inf),
                torch.zeros_like(x)
                )
            )
        sv_square = sigma_y_square - g * sigma_xy_square

        g[sigma_x_square < 0] = 0
        g[sigma_y_square < 0] = 0
        g = g.clamp(min=0)
        sv_square[sigma_x_square < 0] = sigma_y_square[sigma_x_square < 0]
        sv_square[sigma_y_square < 0] = 0
        sv_square = sv_square.clamp(min=0)
        sigma_x_square[sigma_x_square < 0] = 0

        nume += (1 + (g.square() * sigma_x_square).nan_to_num(0) / (sv_square + sigma_nsq)).log10().sum()
        deno += (1 + sigma_x_square / sigma_nsq).log10().sum()

        x = gaussian_blur(x, kernel_size=kernel_size, sigma=sigma) # type: ignore
        y = gaussian_blur(y, kernel_size=kernel_size, sigma=sigma) # type: ignore
        down_sample = MaxPool2d(kernel_size=2, stride=2)
        x: Tensor = down_sample(x)
        y: Tensor = down_sample(y)

    return nume / deno

def calc_VIF(vi: Tensor, ir: Tensor, fu: Tensor):
    VIF = vifp_mscale1(vi, fu) + vifp_mscale1(ir, fu)
    return VIF.item()

def calc_Qabf(vi: Tensor, ir: Tensor, fu: Tensor):
    # [Objective image fusion performance measure](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4883a68d4eea3ef73908d3b1068faf94a615222b)
    L = 1.5
    def quality_of_visual_information(src: Tuple[Tensor, Tensor], dst: Tuple[Tensor, Tensor]):
        gamma_g, gamma_a = 0.9994, 0.9879
        kap_g, kap_a = 15, 22
        sigma_g, sigma_a = 0.5, 0.8
        g_src, a_src = src
        g_dst, a_dst = dst
        g = torch.where(g_src > g_dst, g_dst / g_src, g_src / g_dst).nan_to_num(0)
        a = torch.abs((a_src - a_dst).abs() / (torch.pi / 2) - 1)
        Q_g = gamma_g / (1 + torch.exp(kap_g * (g - sigma_g)))
        Q_a = gamma_a / (1 + torch.exp(kap_a * (a - sigma_a)))
        return Q_g * Q_a

    def sobel_edge(img: Tensor):
        sobel_kernel = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ]).float().unsqueeze(1).to(img.device)
        Sx, Sy = F.conv2d(img.unsqueeze(0), sobel_kernel, padding=1).squeeze(0)
        length = torch.sqrt(Sx.square() + Sy.square())
        alpha = torch.atan((Sy / Sx).nan_to_num(torch.inf, neginf=torch.inf))
        return length, alpha

    edge_vi = sobel_edge(vi)
    edge_ir = sobel_edge(ir)
    edge_fu = sobel_edge(fu)
    Q_vf = quality_of_visual_information(edge_vi, edge_fu)
    Q_if = quality_of_visual_information(edge_ir, edge_fu)
    omega_vi = edge_vi[0].pow(L)
    omega_ir = edge_ir[0].pow(L)
    nume = torch.sum(Q_vf * omega_vi + Q_if * omega_ir)
    deno = torch.sum(omega_vi + omega_ir)
    Qabf = nume / deno
    return Qabf.item()

def calc_SSIM(vi: Tensor, ir: Tensor, fu: Tensor):
    vi, ir, fu = vi.unsqueeze(0), ir.unsqueeze(0), fu.unsqueeze(0)
    return ((ssim(vi, fu, data_range=1.0) + ssim(ir, fu, data_range=1.0)) / 2).item()

def calc_MS_SSIM(vi: Tensor, ir: Tensor, fu: Tensor):
    vi, ir, fu = vi.unsqueeze(0), ir.unsqueeze(0), fu.unsqueeze(0)
    return ((ms_ssim(vi, fu, data_range=1.0) + ms_ssim(ir, fu, data_range=1.0)) / 2).item()

def read_image(path: str) -> Tensor:
    img = io.read_image(path, io.ImageReadMode.GRAY)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    return transform(img)

def calc(vi: Tensor, ir: Tensor, fu: Tensor) -> dict[str, float]:
    return {
        'SF': calc_SF(fu),
        'EN': calc_EN(fu),
        'AG': calc_AG(fu),
        'SD': calc_SD(fu),
        'CE': calc_CE(vi, ir, fu),
        'CC': calc_CC(vi, ir, fu),
        'SCD': calc_SCD(vi, ir, fu),
        'MSE': calc_MSE(vi, ir, fu),
        'MI': calc_MI(vi, ir, fu),
        'PSNR': calc_PSNR(vi, ir, fu),
        'Qabf': calc_Qabf(vi, ir, fu),
        'VIF': calc_VIF(vi, ir, fu),
        'MS-SSIM': calc_MS_SSIM(vi, ir, fu)
    }
