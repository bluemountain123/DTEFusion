from thop import profile, clever_format
import torch
from models.models import FuseInput, DBTFuse, FuseOutput

if __name__ == '__main__':
    model = DBTFuse(
        in_channels=1,
        out_channels=1,
        num_features=32,

        patch_size=16,

        dropout=0.1,
        bias=False
    ).cuda().eval()

    vi=torch.randn(1, 1, 256, 256).cuda()
    ir=torch.randn(1, 1, 256, 256).cuda()


    from thop import profile, clever_format
    FLOPs, Params = profile(model, inputs=(FuseInput(vi, ir),))
    print(FLOPs, Params)
    FLOPs, Params = clever_format([FLOPs * 2, Params], '%.4f')
    print(f'{FLOPs=}, {Params=}')