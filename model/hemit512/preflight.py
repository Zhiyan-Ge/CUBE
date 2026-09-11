import torch
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader

from . import config as cfg
from .dataset import HEMITPKLDataset
from .metrics import batch_pearson
from .network import GANLoss, build_discriminator, build_hemit_generator


def check(name, x):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(cfg.GPU_ID)
    device = torch.device(f"cuda:{cfg.GPU_ID}")

    ds = HEMITPKLDataset(cfg.TRAIN_DIR)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    gen = build_hemit_generator(cfg).to(device).train()
    disc = build_discriminator(cfg).to(device).train()
    l1_fn = torch.nn.L1Loss().to(device)
    gan_fn = GANLoss(cfg.GAN_MODE).to(device)
    opt_g = Adam(gen.parameters(), lr=cfg.LR, betas=(cfg.BETA1, .999))
    opt_d = Adam(disc.parameters(), lr=cfg.LR, betas=(cfg.BETA1, .999))
    sg = GradScaler("cuda", enabled=cfg.USE_AMP)
    sd = GradScaler("cuda", enabled=cfg.USE_AMP)

    print(f"Preflight: 10 real training steps | USE_AMP={cfg.USE_AMP}")
    for step, batch in enumerate(loader, 1):
        he = batch["he"].to(device)
        real = batch["mihc"].to(device)
        check("HE", he); check("target", real)

        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            fake = gen(he)
        check("fake", fake)

        for p in disc.parameters(): p.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            ld = .5 * (
                gan_fn(disc(torch.cat([he, fake.detach()], 1)), False) +
                gan_fn(disc(torch.cat([he, real], 1)), True)
            )
        check("D loss", ld)
        sd.scale(ld).backward(); sd.step(opt_d); sd.update()

        for p in disc.parameters(): p.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=cfg.USE_AMP):
            lgan = gan_fn(disc(torch.cat([he, fake], 1)), True)
            l1 = l1_fn(fake, real)
            lg = lgan + cfg.LAMBDA_L1 * l1
        check("G loss", lg)
        sg.scale(lg).backward(); sg.step(opt_g); sg.update()

        with torch.no_grad():
            corr = batch_pearson(fake, real).mean().item()
        print(f"step={step:02d} D={ld.item():.4f} G={lg.item():.4f} L1={l1.item():.4f} pearson={corr:.4f} fake=[{fake.min().item():.3f},{fake.max().item():.3f}]")
        if step >= 10:
            break

    gen.eval()
    batch = next(iter(DataLoader(HEMITPKLDataset(cfg.VAL_DIR), batch_size=1, shuffle=False, num_workers=0)))
    he = batch["he"].to(device); real = batch["mihc"].to(device)
    with torch.no_grad(), autocast(device_type="cuda", enabled=cfg.USE_AMP):
        pred = gen(he)
    check("validation pred", pred)
    corr = batch_pearson(pred, real)
    print("Validation one-sample Pearson:", corr[0].tolist())
    print("PRELIGHT PASS: safe to start full training.")


if __name__ == "__main__":
    main()
