"""
Self-contained HEMIT dual-branch generator adapted from 1024×1024 to 512×512.

Architecture source:
- Bian et al., HEMIT
- supplied Pix2pix_DualBranch repository, class ResnetGeneratorSwinT
- Swin implementation follows timm 0.4.12 semantics used by that repository

Important adaptation:
    img_size: 1024 -> 512
    patch_size: stays 32
    window_size: stays 64

Keeping patch_size=32 preserves the original scale alignment:
    CNN features: 256, 128, 64
    Swin post-merge features: 8, 4, 2
    five x2 upsampling blocks: 8->256, 4->128, 2->64
"""

import functools
import math

import torch
import torch.nn as nn
from torch.nn import init


# ---------------------------------------------------------------------------
# Generic helpers / pix2pix pieces
# ---------------------------------------------------------------------------


def init_weights(net, init_gain=0.02):
    """Match the supplied HEMIT repository's normal initialization."""

    def init_func(module):
        classname = module.__class__.__name__
        if hasattr(module, "weight") and ("Conv" in classname or "Linear" in classname):
            init.normal_(module.weight.data, 0.0, init_gain)
            if getattr(module, "bias", None) is not None:
                init.constant_(module.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            init.normal_(module.weight.data, 1.0, init_gain)
            init.constant_(module.bias.data, 0.0)

    net.apply(init_func)
    return net


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout=True, use_bias=False):
        super().__init__()
        block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=use_bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            block.append(nn.Dropout(0.5))
        block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=use_bias),
            nn.BatchNorm2d(dim),
        ]
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.conv_block(x)


class NLayerDiscriminator(nn.Module):
    """70x70 PatchGAN used by the HEMIT pix2pix framework."""

    def __init__(self, input_nc=6, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kw,
                    stride=2,
                    padding=padw,
                    bias=False,
                ),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kw,
                stride=1,
                padding=padw,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    def __init__(self, gan_mode="lsgan", target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN mode {gan_mode} is not implemented")

    def forward(self, prediction, target_is_real):
        label = self.real_label if target_is_real else self.fake_label
        return self.loss(prediction, label.expand_as(prediction))


# ---------------------------------------------------------------------------
# Minimal Swin Transformer matching timm 0.4.12 behavior used by HEMIT
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=512, patch_size=32, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(f"Input {(h, w)} does not match configured {self.img_size}")
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(
        b,
        h // window_size,
        window_size,
        w // window_size,
        window_size,
        c,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, c
    )


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(
        b,
        h // window_size,
        w // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        table_size = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=True)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        # Exactly the behavior in timm 0.4.12 used by the HEMIT repository.
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            (self.window_size, self.window_size),
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

        if self.shift_size > 0:
            h, w = self.input_resolution
            img_mask = torch.zeros((1, h, w, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = h_slices
            count = 0
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, hs, ws, :] = count
                    count += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask, persistent=True)

    def forward(self, x):
        h, w = self.input_resolution
        b, length, c = x.shape
        if length != h * w:
            raise ValueError(f"Swin token length {length} != {h}*{w}")

        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = tuple(input_resolution)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, length, c = x.shape
        if length != h * w:
            raise ValueError(f"PatchMerging length {length} != {h}*{w}")
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"PatchMerging requires even size, got {h}x{w}")

        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(b, -1, 4 * c)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio,
        qkv_bias,
        drop,
        attn_drop,
        drop_path,
        downsample,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dp = drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dp,
                )
            )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class MinimalSwinTransformer(nn.Module):
    """Only the Swin pieces actually used by HEMIT's generator."""

    def __init__(
        self,
        img_size=512,
        patch_size=32,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=64,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)
        grid_h, grid_w = self.patch_embed.grid_size

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        offset = 0
        for stage_idx, depth in enumerate(depths):
            resolution = (
                grid_h // (2 ** stage_idx),
                grid_w // (2 ** stage_idx),
            )
            self.layers.append(
                BasicLayer(
                    dim=embed_dim * (2 ** stage_idx),
                    input_resolution=resolution,
                    depth=depth,
                    num_heads=num_heads[stage_idx],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[offset : offset + depth],
                    downsample=stage_idx < len(depths) - 1,
                )
            )
            offset += depth


# ---------------------------------------------------------------------------
# HEMIT Feature Map Fusion and dual-branch generator
# ---------------------------------------------------------------------------


class GatedCrossAttention(nn.Module):
    """
    FMF module from the supplied HEMIT implementation.

    The indexing behavior is intentionally kept compatible with the authors'
    released code instead of silently redesigning the module.
    """

    def __init__(
        self,
        cnn_channels,
        swin_channels,
        num_heads=8,
        k=1000,
        upsample_factor=5,
    ):
        super().__init__()
        self.swin_transform = nn.Conv2d(swin_channels, cnn_channels, kernel_size=1)
        self.attention = nn.MultiheadAttention(embed_dim=cnn_channels, num_heads=num_heads)
        self.gate = nn.Sequential(
            nn.Conv2d(cnn_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.upsample_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        cnn_channels,
                        cnn_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        cnn_channels,
                        cnn_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for _ in range(upsample_factor)
            ]
        )
        self.k = int(k)

    def forward(self, cnn_feature, swin_feature):
        swin_feature = self.swin_transform(swin_feature)
        for upsample_block in self.upsample_blocks:
            swin_feature = upsample_block(swin_feature)

        if swin_feature.shape[-2:] != cnn_feature.shape[-2:]:
            raise RuntimeError(
                f"FMF spatial mismatch: CNN={tuple(cnn_feature.shape)}, "
                f"Swin={tuple(swin_feature.shape)}"
            )

        gate_values = self.gate(cnn_feature)
        b, c, h, w = cnn_feature.shape
        spatial = h * w
        k = min(self.k, spatial)

        cnn_flat = cnn_feature.flatten(2).permute(2, 0, 1)
        swin_flat = swin_feature.flatten(2).permute(2, 0, 1)

        _, top_indices = torch.topk(gate_values.view(b, -1), k=k, dim=1)
        flat_indices = top_indices.reshape(-1)

        cnn_subset = torch.index_select(cnn_flat, 0, flat_indices)
        swin_subset = torch.index_select(swin_flat, 0, flat_indices)

        attended_subset, _ = self.attention(
            cnn_subset,
            swin_subset,
            swin_subset,
            need_weights=False,
        )

        attended = cnn_flat.clone()
        attended.index_copy_(0, flat_indices, attended_subset)
        attended = attended.permute(1, 2, 0).reshape_as(cnn_feature)
        return attended


class HEMITGenerator512(nn.Module):
    """HEMIT ResNet + Swin dual-branch generator for 512×512 inputs."""

    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        ngf=64,
        n_blocks=6,
        use_dropout=True,
        img_size=512,
        patch_size=32,
        window_size=64,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        drop_path_rate=0.2,
        top_k=1000,
    ):
        super().__init__()
        use_bias = False  # BatchNorm2d, matching HEMIT pix2pix defaults

        self.initial_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, padding=0, bias=use_bias),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        self.downsampling_layers = nn.ModuleList()
        for i in range(3):
            mult = 2 ** i
            self.downsampling_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=use_bias,
                    ),
                    nn.BatchNorm2d(ngf * mult * 2),
                    nn.ReLU(True),
                )
            )

        self.resnet_blocks = nn.Sequential(
            *[
                ResnetBlock(ngf * 8, use_dropout=use_dropout, use_bias=use_bias)
                for _ in range(n_blocks)
            ]
        )

        self.swinT = MinimalSwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_nc,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
        )

        upsample_factor = int(math.log2(patch_size))
        self.cross_atts = nn.ModuleList(
            [
                GatedCrossAttention(128, 192, 8, top_k, upsample_factor),
                GatedCrossAttention(256, 384, 8, top_k, upsample_factor),
                GatedCrossAttention(512, 768, 8, top_k, upsample_factor),
            ]
        )

        # Present in the released HEMIT class although unused in forward().
        self.patch_projector = nn.Conv2d(input_nc, embed_dim, kernel_size=1, bias=True)

        self.upsampling_layers = nn.ModuleList()
        for i in range(3):
            mult = 2 ** (3 - i)
            self.upsampling_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        ngf * mult * 2,
                        ngf * mult // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    nn.BatchNorm2d(ngf * mult // 2),
                    nn.ReLU(True),
                )
            )

        self.final_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        cnn = self.initial_layers(x)

        downsampled_features = []
        for down_layer in self.downsampling_layers:
            cnn = down_layer(cnn)
            downsampled_features.append(cnn)

        cnn = self.resnet_blocks(cnn)

        # Swin branch. The released HEMIT code stores post-PatchMerging
        # features from the first three stages and fuses exactly those.
        swin = self.swinT.patch_embed(x)
        swin = self.swinT.pos_drop(swin)
        swin_features = []
        for stage in self.swinT.layers:
            for block in stage.blocks:
                swin = block(swin)
            if stage.downsample is not None:
                swin = stage.downsample(swin)
                hw = math.isqrt(swin.shape[1])
                if hw * hw != swin.shape[1]:
                    raise RuntimeError(f"Non-square Swin token map: {swin.shape}")
                swin_features.append(
                    swin.view(swin.shape[0], hw, hw, swin.shape[2])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

        if len(swin_features) != 3:
            raise RuntimeError(f"Expected 3 Swin fusion maps, got {len(swin_features)}")

        for i, cross_att in enumerate(self.cross_atts):
            downsampled_features[i] = cross_att(
                downsampled_features[i], swin_features[i]
            )

        for up_layer, feature in zip(
            self.upsampling_layers, reversed(downsampled_features)
        ):
            cnn = torch.cat([cnn, feature], dim=1)
            cnn = up_layer(cnn)

        return self.final_layers(cnn)


def build_hemit_generator(cfg):
    net = HEMITGenerator512(
        input_nc=3,
        output_nc=3,
        ngf=cfg.NGF,
        n_blocks=cfg.N_RESBLOCKS,
        use_dropout=cfg.USE_DROPOUT,
        img_size=cfg.IMG_SIZE,
        patch_size=cfg.PATCH_SIZE,
        window_size=cfg.WINDOW_SIZE,
        embed_dim=cfg.EMBED_DIM,
        depths=cfg.DEPTHS,
        num_heads=cfg.NUM_HEADS,
        drop_path_rate=cfg.DROP_PATH_RATE,
        top_k=cfg.TOP_K,
    )
    return init_weights(net, cfg.INIT_GAIN)


def build_discriminator(cfg):
    return init_weights(
        NLayerDiscriminator(input_nc=6, ndf=cfg.NDF, n_layers=3),
        cfg.INIT_GAIN,
    )
