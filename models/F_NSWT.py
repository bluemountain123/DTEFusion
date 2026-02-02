import torch
from torch import Tensor, nn
# from typing import List
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.nn import init
import math

import pywt
import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward
import numpy as np


class F_NSWT(nn.Module):

    def __init__(self, in_channels=32, n=1,
         
                 dct_h=224, dct_w=224,
                 reduction=16, freq_sel_method='top16'):
        super().__init__()
        self.identety = nn.Conv2d(in_channels, in_channels*n, 3, stride=1, padding=1)

        self.DWT = DWTForward(J=3, wave='sym4')

        self.att_low  = MultiSpectralAttentionLayer(
            channel=in_channels,
            dct_h=dct_h, dct_w=dct_w,
            reduction=reduction,
            freq_sel_method=freq_sel_method
        )
        self.att_high = MultiSpectralAttentionLayer(
            channel=in_channels*3,
            dct_h=dct_h, dct_w=dct_w,
            reduction=reduction,
            freq_sel_method=freq_sel_method
        )

        self.dconv_encode = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels*n, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def _transformer(self, low, high):

        return torch.cat([low, high], dim=1)  # 输出 [N, 4C, H, W]


    def forward(self, x):
        # x: [N, C, H, W]
        res = self.identety(x)


        batch, C, H, W = x.shape
        arr = x[0].detach().cpu().numpy() # assume batch=1
        all_cA, all_high = [], []
        wavelet, level = 'sym4', 3

        for ch in range(C):
            cA, (cH, cV, cD) = pywt.swt2(arr[ch], wavelet, level)[0]
            all_cA.append(cA)
            all_high.append(np.stack([cH, cV, cD], axis=0))

        cA_t = torch.from_numpy(np.stack(all_cA, axis=0))\
                    .unsqueeze(0).to(x.device).float()       # [1, C, H, W]
        high_t = torch.from_numpy(np.stack(all_high, axis=0))\
                       .unsqueeze(0).to(x.device).float()   # [1, C, 3, H, W]

        n, C0, K, H0, W0 = high_t.shape
        high_freq = high_t.view(n, C0 * K, H0, W0)    # [1, 3C, H, W]

        # attention
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
        cA_t = cA_t.to(device)
        self.att_low=self.att_low.to(device)
        low_att  = self.att_low(cA_t)  
        self.att_high=self.att_high.to(device)  
        high_freq=high_freq.to(device)  # [1, C, H, W]
        high_att = self.att_high(high_freq)  # [1, 3C, H, W]

        DMT = self._transformer(low_att, high_att)  # [1, 4C, H, W]
        self.dconv_encode=self.dconv_encode.to(device)
        out = self.dconv_encode(DMT)                # [1, C, H, W]
        res = res.to(device)
        out=out + res
        return out


def get_freq_indices(method):#
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:]) 
    if 'top' in method:  
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0,
                             0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6,
                             3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0,
                             1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4,
                             3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4,
                             6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2,
                             2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

# 多频谱注意力层
class MultiSpectralAttentionLayer(nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction# 
        self.dct_h = dct_h#
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        # print("dct_h",dct_h)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
      

        self.dct_layer = MultiSpectralDCTLayer(
            dct_h, dct_w, mapper_x, mapper_y, channel)
        # print("channel",channel,reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((self.dct_h, self.dct_w))

    def forward(self, x):
        # print(x.shape)
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w: 
            x_pooled = self.avgpool(x)
        y = self.dct_layer(x_pooled)
        # print("y在pool",y.shape)
        y = y.float()
        # print("y.float",y.shape)
        y_c = y.shape[1]

        x_in_channels = x.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        conv = nn.Conv2d(in_channels=x_in_channels, out_channels=y_c, kernel_size=1).to(device)
        # print("inchannels",x_in_channels,y_c)
        x=x.to(device)
        x = conv(x)

        y = self.fc(y).view(n, y_c, 1, 1)
        return x * y.expand_as(x)

 # 实现多频谱DCT层
class MultiSpectralDCTLayer(nn.Module):
   

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        # print("channel",channel)
        assert len(mapper_x) == len(mapper_y)  
        assert channel % len(mapper_x) == 0 

        self.num_freq = len(mapper_x) 

        self.weight = self.get_dct_filter(
            height, width, mapper_x, mapper_y, channel)

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + \
                                  str(len(x.shape))

        weight = self.weight.unsqueeze(0)
        in_channels = x.shape[1] 
        out_channels = weight.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1).to(device)
        x=x.to(device)
        x = conv(x)
        
        weight=weight.to(device)
        x = x * weight
        result = torch.sum(torch.sum(x, dim=2), dim=2) 
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros((channel, tile_size_x, tile_size_y))

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(
                        t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter



if __name__ == "__main__":

    in_ch   = 3
    dct_h   = 224
    dct_w   = 224
    block   = F_NSWT(
        in_channels    = in_ch,
        n              = 1,
        dct_h          = dct_h,
        dct_w          = dct_w,
        reduction      = 16,
        freq_sel_method= 'top16'
    )
    block.eval()


    image_path = "/root/autodl-tmp/ESWAcode/00095D_ir.png"  
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((dct_h, dct_w)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    x = transform(image).unsqueeze(0)  

    with torch.no_grad():
        out = block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    out_image = out.squeeze(0).cpu().numpy()  
    out_image = np.transpose(out_image, (1, 2, 0))  
    out_image = np.clip(out_image, 0, 1)  

    output_image = Image.fromarray((out_image * 255).astype(np.uint8))  
    output_image.save("/root/autodl-tmp/data/ir.png") 
    print("Output image saved as 'output_image.jpg'")