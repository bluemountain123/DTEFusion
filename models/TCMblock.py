import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math
from torch.nn import init
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
# Attention
class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,  attn_drop=0., proj_drop=0.):
        """
        :param dim: Dimension of input features
        :param num_heads: Number of attention heads, default is 8
        :param qkv_bias: Whether to use bias in QKV projection, default is False
        :param attn_drop: Dropout probability for attention matrix
        :param proj_drop: Dropout probability for output projection
        """
        super().__init__()  # Call parent class initialization method

        self.heads = num_heads  # Save the number of attention heads

        # Define a Softmax for computing attention weights
        self.attend = nn.Softmax(dim=1)
        # Define a Dropout for randomly dropping attention weights
        self.attn_drop = nn.Dropout(attn_drop)

        # Define a linear layer for generating QKV matrices
        self.qkv = nn.Linear(dim*3, dim*3, bias=qkv_bias)

        # Define a learnable parameter temp for adjusting attention computation
        self.temp = nn.Parameter(torch.ones(12, 1))

        # Define output projection, including a linear layer and a Dropout
        # print("dim",dim)
        self.to_out = nn.Sequential(
            nn.Linear(3*dim, dim*3),  # Linear layer for mapping dimension back to original input dimension
            nn.Dropout(proj_drop)  # Dropout for randomly dropping part of output
        )

    def forward(self, x):
        # Split into multiple heads along the channel dimension
        # [batch_size, seq_length, dim] ===> [batch_size, heads, seq_length, head_dim]
        # torch.Size([1, 784, 64])      ===> torch.Size([1, 8, 784, 8])
        # print("x shape",x.shape)
        is_4d = (x.ndim == 4)
        b1,c1,h1,w1=x.shape
        if is_4d:
            b, c, h, w = x.shape
            x = to_3d(x)          # [B, H*W, C]
        # —— At this point x shape must be [B, N, C] ——

        B, N, C = x.shape
        # print("x",x.shape)
        # print("Number of heads",self.heads)

        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=12)

        # Normalize w along the last dimension, standardization operation
        # torch.Size([1, 8, 784, 8])
        w_normed = torch.nn.functional.normalize(w, dim=-2)
        # Square the normalized w
        w_sq = w_normed ** 2
        # Calculate attention weights Pi: sum w_sq along the last dimension, multiply by temp, then apply Softmax
        # print("w_sq",w_sq.shape)
        # print("temp",(self.temp.unsqueeze(0)).shape)
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)   # Shape is [batch_size, heads, seq_length]

        # Relevant calculation steps of attention operator in the paper
        # This operator: implements low-rank projection, avoids computing pairwise similarities between tokens, has linear computational and memory complexity.
        # Calculate attention scores dots: normalize Pi first, then expand a dimension and multiply by w squared.
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        # Calculate attention matrix attn, formula is 1 / (1 + dots)
        attn = 1. / (1 + dots)
        # Apply dropout to attention matrix to prevent overfitting.
        attn = self.attn_drop(attn)

        # Calculate output, formula is -w * Pi * attn
        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
        # print("out",out.shape)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)  # [B, N, C]
        # print("out1",out.shape)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h1, w=w1)

        return out



## **VSS Block**

class TCM(nn.Module):
    def __init__(
        self,
        
        d_model=16,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        # print("d_mdoel",d_model)
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        # in_ch = x.shape[1] 
        self.pre_conv = nn.LazyConv2d(out_channels=self.d_model,
                              kernel_size=1,
                              bias=conv_bias,
                              **factory_kwargs)
        self.in_norm = nn.LayerNorm(d_model, **factory_kwargs)
        ## Convolution part
        self.branch_conv1 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.branch_conv2 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.act_relu = nn.LeakyReLU()
        # Attention
        self.s2atten = AttentionTSSA(dim=d_model)

        # print("模型")
        # print(self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        # x = x.permute(2, 0, 3, 1)
        # if x.dim() == 4 and x.shape[3] == self.in_ch:
        # print("Before entering ss2D",x.shape)
        # x = x.permute(0, 3, 1, 2).contiguous()
        x = x.float() 
        
        # x = x.permute(0, 1, 3, 2).contiguous()    # → [B, in_ch, H, W]
        # print("Type",type(x))
        # print("Shape input to pre_conv",x.shape)
        # print("Before entering pre_conv",x.shape)
        x = self.pre_conv(x)
        # print("After entering pre_conv",x.shape)                      # → [B, d_model, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()
        # print("x shape before in_norm",x.shape)
        x_ln = self.in_norm(x)
        out1=x_ln
        # print("after out1",out1.shape)
        # Enter convolution layer
        x1 = out1.permute(0, 3, 1, 2)
        branch = self.branch_conv1(x1)
        branch = self.act(branch)
        branch = self.branch_conv2(branch)
        out2 = self.act_relu(branch)
        B, H, W, C = x_ln.shape
        # print("xbeforeINpro",x.shape)
        ## Enter SS2D
        xz = self.in_proj(x_ln)
        # print("xz",xz.shape)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d), split into two parts
        # print("xz",x.shape)
        x = x.permute(0, 3, 1, 2).contiguous()

        # print("x1",x1.shape)
        x = self.act(x) # (b, d, h, w), through activation function Silu
        # print("afterSILU",x.shape)
        y1, y2, y3, y4 = self.forward_core(x) # Through SS2D module
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # print("before_out_normal",y.shape)
        y = self.out_norm(y)

        y = y * F.silu(z) # Perform Hadamard product
        # print("y",y.shape)
        out3 = self.out_proj(y)
        o1 = out1.permute(0, 3, 1, 2)

        o2 = out2
        o3 = out3.permute(0, 3, 1, 2)
    
        fused = torch.cat([o1, o2, o3], dim=1)     
        # print("fused",fused.shape) 
        attn_out = self.s2atten(fused)
        # final = attn_out.permute(0, 2, 3, 1)
        # print("final",final.shape)
        # final = self.out_proj(attn_out)
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)
        return attn_out



## **Model**

### Encoder Block


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choose a d_model (must match your x’s channel dim)
    d_model = 32
    batch_size = 1
    height = 224
    width = 224

    # dummy input: (B, C=d_model, H, W)
    x = torch.randn(batch_size, d_model, height, width, device=device)

    # instantiate SS2D1
    model = TCM(
        d_model=d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=device
    ).to(device)

    # forward
    out = model(x)
    print("input shape:", x.shape)
    print("output shape:", out.shape)

