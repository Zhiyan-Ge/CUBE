"""
Minimal network definitions retained from the supplied HEMIT Pix2pix_DualBranch
repository (MIT licensed):
- ResNet generator
- U-Net generator
- 70×70 PatchGAN discriminator
- LSGAN / vanilla GAN loss

The only architecture adaptation is U-Net depth: num_downs=9 for 512×512 input.
"""

import functools

import torch
import torch.nn as nn
from torch.nn import init


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="batch"):
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda _: Identity()
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


def init_weights(net, init_gain=0.02):
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


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=9,
        padding_type="reflect",
    ):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        for _ in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * 4,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    ngf * mult // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self._build(dim, padding_type, norm_layer, use_dropout, use_bias)

    @staticmethod
    def _padding(padding_type):
        if padding_type == "reflect":
            return [nn.ReflectionPad2d(1)], 0
        if padding_type == "replicate":
            return [nn.ReplicationPad2d(1)], 0
        if padding_type == "zero":
            return [], 1
        raise NotImplementedError(f"padding [{padding_type}] is not implemented")

    def _build(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        first_pad, p = self._padding(padding_type)
        block = first_pad + [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            block += [nn.Dropout(0.5)]

        second_pad, p = self._padding(padding_type)
        block += second_pad + [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        return nn.Sequential(*block)

    def forward(self, x):
        return x + self.conv_block(x)


class UnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        num_downs=9,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()

        block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            innermost=True,
            norm_layer=norm_layer,
        )
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                submodule=block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, x):
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
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
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class GANLoss(nn.Module):
    def __init__(self, gan_mode="lsgan", target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"gan mode [{gan_mode}] is not implemented")

    def forward(self, prediction, target_is_real):
        label = self.real_label if target_is_real else self.fake_label
        return self.loss(prediction, label.expand_as(prediction))


def build_generator(kind, ngf=64, norm="batch", use_dropout=False, init_gain=0.02):
    norm_layer = get_norm_layer(norm)
    if kind == "unet":
        net = UnetGenerator(3, 3, num_downs=9, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif kind == "resnet":
        net = ResnetGenerator(3, 3, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    else:
        raise ValueError(f"Unknown generator kind: {kind}")
    return init_weights(net, init_gain)


def build_discriminator(ndf=64, norm="batch", init_gain=0.02):
    norm_layer = get_norm_layer(norm)
    net = NLayerDiscriminator(6, ndf=ndf, n_layers=3, norm_layer=norm_layer)
    return init_weights(net, init_gain)
