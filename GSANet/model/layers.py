import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

from model.cvca import multihead_geometric_transform_attention, multihead_vecrep_attention
import numpy as np

import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
__USE_DEFAULT_INIT__ = False

class JaxLinear(nn.Linear):
    """ Linear layers with initialization matching the Jax defaults """

    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            input_size = self.weight.shape[-1]
            std = math.sqrt(1/input_size)
            init.trunc_normal_(self.weight, std=std, a=-2.*std, b=2.*std)
            if self.bias is not None:
                init.zeros_(self.bias)


class ViTLinear(nn.Linear):
    """ Initialization for linear layers used by ViT """

    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.normal_(self.bias, std=1e-6)


class SRTLinear(nn.Linear):
    """ Initialization for linear layers used in the SRT decoder """

    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)


class PositionalEncoding(nn.Module):

    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

        octaves = torch.arange(
            self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float()
        self.multipliers = 2**octaves * math.pi

    def forward(self, coords):

        shape, dim = coords.shape[:-1], coords.shape[-1]

        multipliers = self.multipliers.to(coords)
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(
            *shape, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(
            *shape, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class RayPosEncoder(nn.Module):
    def __init__(self, pos_octaves=8, pos_start_octave=0, ray_octaves=4, ray_start_octave=0):
        super().__init__()
        self.pos_encoding = PositionalEncoding(
            num_octaves=pos_octaves, start_octave=pos_start_octave)
        self.ray_encoding = PositionalEncoding(
            num_octaves=ray_octaves, start_octave=ray_start_octave)

    def forward(self, pos, rays):
        pos_enc = self.pos_encoding(pos)
        ray_enc = self.ray_encoding(rays)
        x = torch.cat((pos_enc, ray_enc), -1)
        return x


class RayOnlyEncoder(nn.Module):
    def __init__(self, ray_octaves=4, ray_start_octave=0):
        super().__init__()
        self.ray_encoding = PositionalEncoding(
            num_octaves=ray_octaves, start_octave=ray_start_octave)

    def forward(self, rays):
        if len(rays.shape) == 4:
            batchsize, height, width, dims = rays.shape
            rays = rays.flatten(1, 2)
            ray_enc = self.ray_encoding(rays)
            ray_enc = ray_enc.view(batchsize, height, width, ray_enc.shape[-1])
            ray_enc = ray_enc.permute((0, 3, 1, 2))
        else:
            ray_enc = self.ray_encoding(rays)

        return ray_enc


class LearnedRayEmbedding(nn.Module):
    def __init__(self, ray_octaves=60, ray_start_octave=-30, h=320, w=240):
        super().__init__()
        self.encoding = PositionalEncoding(
            num_octaves=ray_octaves, start_octave=ray_start_octave)
        initial_emb = self.encoding(get_vertical_rays(width=w, height=h)[None])
        self.emb = nn.Parameter(initial_emb)
        print(self.emb.shape)

    def forward(self, N):
        return self.emb[None].repeat(N, 1, 1, 1)


# Transformer implementation based on ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py


class TemperatureAdjsutableSoftmax(nn.Module):

    def __init__(self, init_tau=1.0, dim=-1):
        super().__init__()
        self.tau = nn.Parameter(torch.Tensor([init_tau]))
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x/self.tau)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(
            dim) if dim is not None else lambda x: torch.nn.functional.normalize(x, dim=-1)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., linear_module=ViTLinear):
        super().__init__()
        self.net = nn.Sequential(
            linear_module(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            linear_module(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)

def transform_tensor(x, b, head, h, w, c, p):
    # 输入 x: [b, head, 5*h*w, c/head]
    x = rearrange(x, 'b head n (c_head) -> b n (head c_head)', head=head) # [b, 5*h*w, c]
    x = rearrange(x, 'b (v hw) c -> b v hw c', v=5) # [b, 5, h*w, c]
    x = rearrange(x, 'b v hw c -> (b v) hw c') # [b*5, h*w, c]
    x = rearrange(x, 'b (hw m) c -> b (hw) (c m)', m=p)  # [b*5, h*w//4, c*4]
    x = rearrange(x, 'b hw (head c_head) -> b head hw c_head', head=head) 
    return x  # 输出: [b*5, head, h*w//4, (c*4)//head]
    
def transform_tensor_1(x, b, head, h, w, c, p):
    # 输入 x: [b*5, head, h*w//4, (c*4)//head] 
    x = rearrange(x, 'b head hw c_head -> b hw (head c_head)', head=head) #[b*5, h*w//4, c*4]
    x = rearrange(x, 'b (hw) (c m) -> b (hw m) c', m=p)  # [b*5, h*w, c]
    x = rearrange(x, '(b v) hw c -> b v hw c', v=5) # [b, 5, h*w, c]
    x = rearrange(x, 'b v hw c -> b (v hw) c') # [b, 5*h*w, c]
    x = rearrange(x, 'b n (head c_head) -> b head n (c_head)', head=head) #[b, head, 5*h*w, c/head]
    return x # 输出: [b, head, 5*h*w, c/head]

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kv_dim=None, attn_args={}, linear_module=JaxLinear, **kwargs):
        super().__init__()
        inner_dim = dim_head * heads # 64
        project_out = not (heads == 1 and dim_head == dim)

        # kv_dim being None indicates this attention is used as self-attention
        self.selfatt = (kv_dim is None)
        self.heads = heads
        self.scale = dim_head ** -0.5
            
        self.trans_coeff = nn.Parameter(torch.Tensor([0.01]))

        # self.attend = nn.Softmax(-1)
        tau = 1.0

        class AttnFn(torch.nn.Module):
            def __init__(self, scale):
                self.scale = scale
                super().__init__()

            def forward(self, q, k, v):
                b, h, n, d = q.shape
                if n == 4090 or n == 5120 or n == 3072 or n == 2048: # 5120
                    q = q.reshape(b, h, -1, d*4)
                    k = k.reshape(b, h, -1, d*4)
                    v = v.reshape(b, h, -1, d*4)
                    # height, width = 32, 32
                    # q = transform_tensor(q, b, h, height, width, d*h, p=4)
                    # k = transform_tensor(k, b, h, height, width, d*h, p=4)
                    # v = transform_tensor(v, b, h, height, width, d*h, p=4)
                if n == 16384 or n == 20480 or n == 12288 or n == 8192: #20480
                    q = q.reshape(b, h, -1, d*16)
                    k = k.reshape(b, h, -1, d*16)
                    v = v.reshape(b, h, -1, d*16)
                    # height, width = 64, 64
                    # q = transform_tensor(q, b, h, height, width, d*h, p=16)
                    # k = transform_tensor(k, b, h, height, width, d*h, p=16)
                    # v = transform_tensor(v, b, h, height, width, d*h, p=16)
                sim = q @ k.transpose(-1, -2)  # [B, H, Nq*Tq, Nk*Tk]
                self.scale = k.shape[-1] ** -0.5
                attn = nn.Softmax(-1)(sim * self.scale / tau)
                out = (attn @ v)
                # if n == 5120:
                #     out = transform_tensor_1(q, b, h, height, width, d*h, p=4)
                # if n == 20480:
                #     out = transform_tensor_1(q, b, h, height, width, d*h, p=16)
                out = out.reshape(b,h,n,d)
                return out, attn

        class EuclidAttnFn(torch.nn.Module):
            def __init__(self, scale):
                self.scale = scale
                super().__init__()
                print("""Euclid Attention""")

            def forward(self, q, k, v):
                # sim(Q, K) = -0.5*||Q-K||^2 = Q'K - 0.5Q'Q - 0.5K'K
                sim = q @ k.transpose(-1, -2)  - 0.5 * q.pow(2).sum(-1)[..., None] - 0.5 * k.pow(2).sum(-1)[..., None, :]
                attn = nn.Softmax(-1)(sim * self.scale / tau)
                out = (attn @ v)
                return out, attn

        self.euclid = False
        self.attn_fn = EuclidAttnFn(self.scale) if self.euclid else AttnFn(self.scale)
        self.use_bias = False
        # parse
        q_inner_dim = inner_dim
        
        # self.to_q = linear_module(dim, q_inner_dim, bias=self.use_bias)
        # self.to_kv = linear_module(
        #     dim, 2*inner_dim, bias=self.use_bias)
        
        self.to_qkv = linear_module(
            dim, inner_dim * 2 + q_inner_dim, bias=self.use_bias)


        self.to_out = nn.Sequential(
            linear_module(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        # self.to_out = nn.Sequential(
        #     linear_module(inner_dim if not self.rpe else inner_dim +
        #               self.heads*self.q_bias.shape[-1], dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()
        self.f_dims = attn_args['f_dims']

    def forward(self, x, z=None, return_attmap=False, extras={}):

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        cross = False
        # else:
        #     q = self.to_q(x)
        #     k, v = self.to_kv(z).chunk(2, dim=-1)
        #     qkv = (q, k, v)      
        #     cross = True
        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)


        # (extras['vecrep_q'],
        #     extras['vecrep_k'],
        #     extras['vecinvrep_q']) = map(lambda frep: self.rep_to_vec(frep), 
        #                                         (extras['flattened_rep_q'],
        #                                         extras['flattened_rep_k'],
        #                                         extras['flattened_invrep_q'])) 
        # fn = multihead_vecrep_attention
     
        fn = multihead_geometric_transform_attention

        v_transform = True
        out = fn(
            q, k, v, attn_fn=self.attn_fn,
            f_dims=self.f_dims,
            reps=extras,
            trans_coeff=self.trans_coeff,
            v_transform=v_transform,
            euclid=self.euclid,
            cross=cross)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
            

        # if return_attmap:
        #     return out, attn
        # else:
        #     return out
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.,
                 selfatt=True,
                 kv_dim=None,
                 return_last_attmap=False,
                 attn_args={}):

        super().__init__()
        self.heads = heads
        self.layers = nn.ModuleList([])

        linear_module_attn = lambda *args, **kwargs: JaxLinear(*args, **kwargs)
        linear_module_ff = lambda *args, **kwargs: ViTLinear(*args, **kwargs)

        prenorm_fn = lambda m: PreNorm(dim, m)
        # prenorm_fn = lambda m: PreNorm(72, m)
        for k in range(depth):
            attn = prenorm_fn(Attention(
                    dim, heads=heads, dim_head=dim_head,
                    dropout=dropout, selfatt=selfatt, kv_dim=kv_dim, attn_args=attn_args, 
                    linear_module=linear_module_attn))
            ff = prenorm_fn(FeedForward(
                dim, mlp_dim,
                dropout=dropout,
                linear_module=linear_module_ff))
            self.layers.append(nn.ModuleList([attn, ff]))
        self.return_last_attmap = return_last_attmap

    def forward(self, x, z=None, extras=None):
        
        for l, (attn, ff) in enumerate(self.layers):
            if l == len(self.layers)-1 and self.return_last_attmap:
                out = attn(x, z=z, return_attmap=True, extras=extras)
                x = out + x
            else:
                x = attn(x, z=z, extras=extras) + x
            x = ff(x) + x

        # if self.return_last_attmap:
        #     return x, attmap
        # else:
        #     return x
        return x
 
class Attention_Dec(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kv_dim=None, attn_args={}, linear_module=JaxLinear, **kwargs):
        super().__init__()
        inner_dim = dim_head * heads # 64
        project_out = not (heads == 1 and dim_head == dim)

        # kv_dim being None indicates this attention is used as self-attention
        self.selfatt = (kv_dim is None)
        self.heads = heads
        self.scale = dim_head ** -0.5


        
        self.trans_coeff = nn.Parameter(torch.Tensor([0.01]))

        # self.attend = nn.Softmax(-1)
        tau = 1.0

        class AttnFn(torch.nn.Module):
            def __init__(self, scale):
                self.scale = scale
                super().__init__()

            def forward(self, q, k, v):
                b, h, n, d = q.shape
                if n == 5120:
                    q = q.reshape(b, h, -1, d*4)
                    k = k.reshape(b, h, -1, d*4)
                    v = v.reshape(b, h, -1, d*4)
                    # height, width = 32, 32
                    # q = transform_tensor(q, b, h, height, width, d*h, p=4)
                    # k = transform_tensor(k, b, h, height, width, d*h, p=4)
                    # v = transform_tensor(v, b, h, height, width, d*h, p=4)
                if n == 20480:
                    q = q.reshape(b, h, -1, d*16)
                    k = k.reshape(b, h, -1, d*16)
                    v = v.reshape(b, h, -1, d*16)
                    # height, width = 64, 64
                    # q = transform_tensor(q, b, h, height, width, d*h, p=16)
                    # k = transform_tensor(k, b, h, height, width, d*h, p=16)
                    # v = transform_tensor(v, b, h, height, width, d*h, p=16)
                sim = q @ k.transpose(-1, -2)  # [B, H, Nq*Tq, Nk*Tk]
                self.scale = k.shape[-1] ** -0.5
                attn = nn.Softmax(-1)(sim * self.scale / tau)
                out = (attn @ v)
                # if n == 5120:
                #     out = transform_tensor_1(q, b, h, height, width, d*h, p=4)
                # if n == 20480:
                #     out = transform_tensor_1(q, b, h, height, width, d*h, p=16)
                out = out.reshape(b,h,n,d)
                return out, attn

        class EuclidAttnFn(torch.nn.Module):
            def __init__(self, scale):
                self.scale = scale
                super().__init__()
                print("""Euclid Attention""")

            def forward(self, q, k, v):
                # sim(Q, K) = -0.5*||Q-K||^2 = Q'K - 0.5Q'Q - 0.5K'K
                sim = q @ k.transpose(-1, -2)  - 0.5 * q.pow(2).sum(-1)[..., None] - 0.5 * k.pow(2).sum(-1)[..., None, :]
                attn = nn.Softmax(-1)(sim * self.scale / tau)
                out = (attn @ v)
                return out, attn

        self.euclid = False
        self.attn_fn = EuclidAttnFn(self.scale) if self.euclid else AttnFn(self.scale)
        self.use_bias = False
        # parse
        q_inner_dim = inner_dim
        
        self.to_q = linear_module(dim, q_inner_dim, bias=self.use_bias)
        self.to_kv = linear_module(
            dim, 2*inner_dim, bias=self.use_bias)
        
        # self.to_qkv = linear_module(
        #     dim, inner_dim * 2 + q_inner_dim, bias=self.use_bias)


        self.to_out = nn.Sequential(
            linear_module(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        # self.to_out = nn.Sequential(
        #     linear_module(inner_dim if not self.rpe else inner_dim +
        #               self.heads*self.q_bias.shape[-1], dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()
        self.f_dims = attn_args['f_dims']

    def forward(self, x, z=None, return_attmap=False, extras={}):

        
        q = self.to_q(x)
        k, v = self.to_kv(z).chunk(2, dim=-1)
        qkv = (q, k, v)      
        cross = True
        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)


        # (extras['vecrep_q'],
        #     extras['vecrep_k'],
        #     extras['vecinvrep_q']) = map(lambda frep: self.rep_to_vec(frep), 
        #                                         (extras['flattened_rep_q'],
        #                                         extras['flattened_rep_k'],
        #                                         extras['flattened_invrep_q'])) 
        # fn = multihead_vecrep_attention
     
        fn = multihead_geometric_transform_attention

        v_transform = True
        out = fn(
            q, k, v, attn_fn=self.attn_fn,
            f_dims=self.f_dims,
            reps=extras,
            trans_coeff=self.trans_coeff,
            v_transform=v_transform,
            euclid=self.euclid,
            cross=cross)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
            

        # if return_attmap:
        #     return out, attn
        # else:
        #     return out
        return out


class Transformer_Dec(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout=0.,
                 selfatt=True,
                 kv_dim=None,
                 return_last_attmap=False,
                 attn_args={}):

        super().__init__()
        self.heads = heads
        self.layers = nn.ModuleList([])

        linear_module_attn = lambda *args, **kwargs: JaxLinear(*args, **kwargs)
        linear_module_ff = lambda *args, **kwargs: ViTLinear(*args, **kwargs)

        prenorm_fn = lambda m: PreNorm(dim, m)
        for k in range(depth):
            attn = prenorm_fn(Attention_Dec(
                    dim, heads=heads, dim_head=dim_head,
                    dropout=dropout, selfatt=selfatt, kv_dim=kv_dim, attn_args=attn_args, 
                    linear_module=linear_module_attn))
            ff = prenorm_fn(FeedForward(
                dim, mlp_dim,
                dropout=dropout,
                linear_module=linear_module_ff))
            self.layers.append(nn.ModuleList([attn, ff]))
        self.return_last_attmap = return_last_attmap

    def forward(self, x, z=None, extras=None):
        
        for l, (attn, ff) in enumerate(self.layers):
            if l == len(self.layers)-1 and self.return_last_attmap:
                out = attn(x, z=z, return_attmap=True, extras=extras)
                x = out + x
            else:
                x = attn(x, z=z, extras=extras) + x
            x = ff(x) + x

        # if self.return_last_attmap:
        #     return x, attmap
        # else:
        #     return x
        return x