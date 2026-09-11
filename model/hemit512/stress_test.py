"""Long preflight for HEMIT-512.

Runs enough real optimization steps to cover the point where the previous
k=1000 adaptation failed (~step 875), while checking actual tensors/gradients.
No checkpoint from this script should be used as a benchmark result.
"""
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from . import config as cfg
from .dataset import HEMITPKLDataset
from .network import GANLoss, build_discriminator, build_hemit_generator
from .trainer import assert_finite, assert_finite_gradients, set_requires_grad

STEPS = 1100


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    seed_everything(cfg.SEED)
    torch.cuda.set_device(cfg.GPU_ID)
    device = torch.device(f"cuda:{cfg.GPU_ID}")

    ds = HEMITPKLDataset(cfg.TRAIN_DIR)
    loader = DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.SEED),
    )

    gen = build_hemit_generator(cfg).to(device).train()
    disc = build_discriminator(cfg).to(device).train()
    l1_fn = torch.nn.L1Loss().to(device)
    gan_fn = GANLoss(cfg.GAN_MODE).to(device)
    opt_g = Adam(gen.parameters(), lr=cfg.LR, betas=(cfg.BETA1, .999))
    opt_d = Adam(disc.parameters(), lr=cfg.LR, betas=(cfg.BETA1, .999))

    print(f"Stress test: {STEPS} steps | FP32 | TOP_K={cfg.TOP_K} | batch={cfg.BATCH_SIZE}")
    iterator = iter(loader)
    for step in range(1, STEPS + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        he = batch["he"].to(device, non_blocking=True)
        real = batch["mihc"].to(device, non_blocking=True)
        names = list(batch["sample_name"])
        assert_finite("HE input", he)
        assert_finite("mIHC target", real)

        fake = gen(he)
        assert_finite("generator output", fake)

        set_requires_grad(disc, True)
        opt_d.zero_grad(set_to_none=True)
        fake_pair_d = torch.cat([he, fake.detach()], 1)
        real_pair = torch.cat([he, real], 1)
        ld = .5 * (gan_fn(disc(fake_pair_d), False) + gan_fn(disc(real_pair), True))
        assert_finite("D loss", ld)
        ld.backward()
        try:
            assert_finite_gradients(disc, "discriminator")
        except Exception:
            print(f"FAILED D at step {step}; samples={names}")
            raise
        opt_d.step()

        set_requires_grad(disc, False)
        opt_g.zero_grad(set_to_none=True)
        lgan = gan_fn(disc(torch.cat([he, fake], 1)), True)
        l1 = l1_fn(fake, real)
        lg = lgan + cfg.LAMBDA_L1 * l1
        assert_finite("G GAN loss", lgan)
        assert_finite("G L1 loss", l1)
        assert_finite("G total loss", lg)
        lg.backward()
        try:
            assert_finite_gradients(gen, "generator")
        except Exception:
            print(f"FAILED G at step {step}; samples={names}")
            print(f"HE range=[{he.min().item():.6f},{he.max().item():.6f}] target range=[{real.min().item():.6f},{real.max().item():.6f}]")
            print(f"fake range=[{fake.min().item():.6f},{fake.max().item():.6f}] D={ld.item():.6f} G={lg.item():.6f} GAN={lgan.item():.6f} L1={l1.item():.6f}")
            raise
        opt_g.step()

        if step == 1 or step % 100 == 0:
            print(
                f"step={step:04d} D={ld.item():.4f} G={lg.item():.4f} "
                f"GAN={lgan.item():.4f} L1={l1.item():.4f} "
                f"fake=[{fake.min().item():.3f},{fake.max().item():.3f}]"
            )

    print("STRESS TEST PASS: 1100 optimization steps completed with finite outputs, losses, and individual gradients.")


if __name__ == "__main__":
    main()
