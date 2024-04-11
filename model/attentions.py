# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from functools import partial


class GlobalAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim, num_heads=4, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalTransformer(nn.Module):
    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6), qkv_bias=False, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = GlobalAttention(dim, num_heads=4, qkv_bias=qkv_bias,
                                     qk_scale=qk_scale, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        self.attn2 = GlobalAttention(dim, num_heads=4, qkv_bias=qkv_bias,
                                     qk_scale=qk_scale, attn_drop=attn_drop)
        self.conv = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, x):
        x_1 = x.permute(0, 2, 3, 1)
        x_1 = x_1 + self.attn1(self.norm1(x_1))
        x_1 = x_1.permute(0, 3, 1, 2)

        x_2 = x_1.permute(0, 2, 3, 1)
        x_2 = x_2 + self.attn2(self.norm2(x_2))
        x_2 = x_2.permute(0, 3, 1, 2)
        out = self.conv(torch.cat((x, x_1, x_2), dim=1))

        return out
