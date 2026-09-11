import torch
from torch.amp import GradScaler, autocast

from . import config as cfg
from .network import GANLoss, build_discriminator, build_hemit_generator


def check(name, x):
    ok = torch.isfinite(x).all().item()
    print(f"{name}: shape={tuple(x.shape) if hasattr(x, 'shape') else ()} finite={ok}")
    if not ok:
        raise FloatingPointError(f"{name} contains NaN/Inf")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the exact HEMIT-512 smoke test.")

    torch.cuda.set_device(cfg.GPU_ID)
    device = torch.device(f"cuda:{cfg.GPU_ID}")

    # Use the exact initialized builders used by trainer.py.
    generator = build_hemit_generator(cfg).to(device).train()
    discriminator = build_discriminator(cfg).to(device).train()
    gan_loss = GANLoss(cfg.GAN_MODE).to(device)
    l1_loss = torch.nn.L1Loss().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.LR, betas=(cfg.BETA1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.LR, betas=(cfg.BETA1, 0.999))
    scaler_g = GradScaler("cuda", enabled=cfg.USE_AMP)
    scaler_d = GradScaler("cuda", enabled=cfg.USE_AMP)

    he = torch.rand(cfg.BATCH_SIZE, 3, 512, 512, device=device).mul_(2).sub_(1)
    real = torch.rand(cfg.BATCH_SIZE, 3, 512, 512, device=device).mul_(2).sub_(1)

    with autocast(device_type="cuda", enabled=cfg.USE_AMP):
        fake = generator(he)
    check("generator output", fake)

    opt_d.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", enabled=cfg.USE_AMP):
        d_fake = gan_loss(discriminator(torch.cat([he, fake.detach()], 1)), False)
        d_real = gan_loss(discriminator(torch.cat([he, real], 1)), True)
        loss_d = 0.5 * (d_fake + d_real)
    check("D loss", loss_d)
    scaler_d.scale(loss_d).backward()
    scaler_d.step(opt_d)
    scaler_d.update()

    for p in discriminator.parameters():
        p.requires_grad_(False)
    opt_g.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", enabled=cfg.USE_AMP):
        loss_gan = gan_loss(discriminator(torch.cat([he, fake], 1)), True)
        loss_l1 = l1_loss(fake, real)
        loss_g = loss_gan + cfg.LAMBDA_L1 * loss_l1
    check("G GAN loss", loss_gan)
    check("G L1 loss", loss_l1)
    check("G total loss", loss_g)
    scaler_g.scale(loss_g).backward()
    scaler_g.step(opt_g)
    scaler_g.update()

    generator.eval()
    with torch.no_grad(), autocast(device_type="cuda", enabled=cfg.USE_AMP):
        y = generator(he[:1])
    check("post-step eval output", y)
    print(f"PASS | USE_AMP={cfg.USE_AMP} | output range=({y.min().item():.4f}, {y.max().item():.4f})")


if __name__ == "__main__":
    main()
