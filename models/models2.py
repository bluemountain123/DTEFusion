from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.fusionlayer import ModularNetwork
from models.F_NSWT import F_NSWT
from models.TCMblock import TCM

@dataclass
class FuseInput:
    vi: Tensor
    ir: Tensor

@dataclass
class FuseOutput:
    fusion: Tensor
    ir_out: Tensor
    vi_out: Tensor


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels: int, mid_channels: list[int]):
        super().__init__()
        assert len(mid_channels) == 2, "mid_channels list must contain exactly two elements for the two intermediate convolution layers."

        self.conv1 = nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_channels[1], 1, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # print("beforeDecoder",x.shape)
        x = self.conv1(x)
        # print("decoder1",x.shape)
        x = self.relu1(x)
        x = self.conv2(x)
        # print("decoder2",x.shape)
        x = self.relu2(x)
        x = self.conv3(x)
        # print("decoder3",x.shape)
        x = self.tanh(x)
        return x

class CAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_feat, text_feat):
        B, C, H, W = img_feat.shape

        q = img_feat.view(B, C, -1)
        k = text_feat.view(B, C, -1).permute(0, 2, 1)
        attention_map = torch.bmm(q, k)
        attention_map = self.softmax(attention_map)

        v = text_feat.view(B, C, -1)
        attention_info = torch.bmm(attention_map, v)
        attention_info = attention_info.view(B, C, H, W)
        output = self.gamma * attention_info + img_feat
        return output

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

class Frequency_Domain(nn.Module):
    def __init__(self, channels=32):
        super().__init__()

        self.norm = LayerNorm2d(channels)
        self.freq = FreMLP(nc=channels, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        A = inp
        x_step2 = self.norm(inp)  
        x_freq = self.freq(x_step2) 
        x = A * x_freq
        x = A + x * self.gamma
        return x
    
class DBTFuse1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int,

        patch_size: 16
    ):
        super().__init__()
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge=F_NSWT()

        self.freq_domain_vi = Frequency_Domain()
        self.Mam = TCM()
        self.conv = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding='same', dilation=1, groups=in_channels, bias=True, padding_mode='zeros')
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.ln_vi = nn.InstanceNorm2d(4)
        self.ln_ir = nn.InstanceNorm2d(4)
        self.cm= CAM()
        self.conv_ir = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        decoder_channels = [num_features + num_features, 32, 16] 
        self.decoder_vi = SimpleDecoder(num_features, decoder_channels[1:])
        self.decoder_ir = SimpleDecoder(num_features, decoder_channels[1:])
        self.decoder_fuse = SimpleDecoder(48, decoder_channels[out_channels:])
        self.fusion_strategy = ModularNetwork(num_features, num_blocks=1, kernel_size=3, stride=1)

        self.proj_out = nn.Tanh()


    def forward(self, inputs: FuseInput):

        ir_N1 = self.conv_ir(inputs.ir)  
        
        vi_N1 = self.conv_ir(inputs.vi)  
        vi_N=self.edge(vi_N1)

        
        ir_S = self.freq_domain_vi(ir_N1)
        vi_N = vi_N1+vi_N

        vi_M = self.Mam(vi_N)
        ir_M = self.Mam(ir_S)


        fused = self.cm(ir_M,vi_M)

        merged = self.fusion_strategy(fused)
        r_ir = self.proj_out(self.decoder_ir(ir_M))
        r_vi = self.proj_out(self.decoder_vi(vi_M))
        fusion = self.proj_out(self.decoder_fuse(merged))
    
        return FuseOutput(
            fusion=fusion.clamp(-1.0, 1.0),
            vi_out=r_vi.clamp(-1.0, 1.0),
            ir_out=r_ir.clamp(-1.0, 1.0)
        )

    def frozen(self, *__names: str):
        for name, param in self.named_parameters():
            for _name in __names:
                if _name in name:
                    param.requires_grad = False
                    print('frozen layer:', name)

if __name__ == '__main__':
    model = DBTFuse1(
        in_channels=1,
        out_channels=1,
        num_features=48,
        patch_size= 16

    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    vi = torch.randn(1,1, 224, 224).to(device)  # input VIS,batchsize=3,channel=3
    ir = torch.randn(1, 1,224, 224).to(device)  # input IR

    fuse_input = FuseInput(vi=vi, ir=ir)

    with torch.no_grad():  
        fuse_output = model(fuse_input)


    print(f"Fusion output shape: {fuse_output.fusion.shape}")
