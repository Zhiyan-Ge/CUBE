import torch
import torch.nn as nn

from . import config as cfg
from .networks import GANLoss, build_discriminator, build_generator


class BenchmarkModel(nn.Module):
    """One compact wrapper for U-Net, ResNet and their pix2pix variants."""

    VALID_BENCHMARKS = {
        "unet": ("unet", False),
        "resnet": ("resnet", False),
        "pix2pix_unet": ("unet", True),
        "pix2pix_resnet": ("resnet", True),
    }

    def __init__(self, benchmark: str):
        super().__init__()
        if benchmark not in self.VALID_BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {benchmark}")

        generator_kind, use_gan = self.VALID_BENCHMARKS[benchmark]
        self.benchmark = benchmark
        self.use_gan = use_gan

        use_dropout = cfg.USE_DROPOUT if use_gan else False

        self.generator = build_generator(
            generator_kind,
            ngf=cfg.NGF,
            norm=cfg.NORM,
            use_dropout=use_dropout,
            init_gain=cfg.INIT_GAIN,
        )
        self.discriminator = None
        if use_gan:
            self.discriminator = build_discriminator(
                ndf=cfg.NDF,
                norm=cfg.NORM,
                init_gain=cfg.INIT_GAIN,
            )

    def forward(self, he):
        return self.generator(he)


def build_losses(device):
    return {
        "l1": nn.L1Loss().to(device),
        "gan": GANLoss(cfg.GAN_MODE).to(device),
    }


def discriminator_loss(model, losses, he, real_mihc, fake_mihc):
    fake_pair = torch.cat([he, fake_mihc.detach()], dim=1)
    real_pair = torch.cat([he, real_mihc], dim=1)
    loss_fake = losses["gan"](model.discriminator(fake_pair), False)
    loss_real = losses["gan"](model.discriminator(real_pair), True)
    return 0.5 * (loss_fake + loss_real)


def generator_loss(model, losses, he, real_mihc, fake_mihc):
    l1 = losses["l1"](fake_mihc, real_mihc)
    if not model.use_gan:
        return l1, {"l1": l1.detach(), "gan": torch.zeros((), device=l1.device)}

    fake_pair = torch.cat([he, fake_mihc], dim=1)
    gan = losses["gan"](model.discriminator(fake_pair), True)
    total = gan + cfg.LAMBDA_L1 * l1
    return total, {"l1": l1.detach(), "gan": gan.detach()}
