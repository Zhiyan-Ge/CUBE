import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from .dataset import HEMIHCDataset
from .model import HEMIHCBranch
from he_mihc_shared_decoder import config as cfg


N_SAMPLES = 8

CHECKPOINTS = [
    ("epoch10", "epoch_10.pt"),
    ("epoch20", "epoch_20.pt"),
    ("epoch30", "epoch_30.pt"),
    ("epoch50", "epoch_50.pt"),
    ("best", "best.pt"),
    ("last", "last.pt"),
]


def to_rgb(x):
    x = x.detach().float().cpu()
    x = x.permute(1, 2, 0).numpy()
    return np.clip(x, 0, 1)


def load_model(path, device):
    model = HEMIHCBranch(cfg.GROUPS).to(device)

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    model.eval()
    return model


def main():
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(cfg.OUTPUT_DIR, "diagnostics", "reconstruction")
    os.makedirs(save_dir, exist_ok=True)

    dataset = HEMIHCDataset(cfg.VAL_DIR)

    # Use the same validation samples for every checkpoint.
    generator = torch.Generator().manual_seed(cfg.SEED)
    indices = torch.randperm(len(dataset), generator=generator)[:N_SAMPLES].tolist()

    samples = [dataset[i] for i in indices]

    # Run inference for all available checkpoints.
    predictions = {}

    for label, filename in CHECKPOINTS:
        path = os.path.join(cfg.OUTPUT_DIR, filename)

        if not os.path.exists(path):
            continue

        print(f"Loading {filename}")

        model = load_model(path, device)
        predictions[label] = {}

        with torch.inference_mode():
            for sample in samples:
                he = sample["he"].unsqueeze(0).to(device)
                mihc = sample["mihc"].unsqueeze(0).to(device)

                with torch.amp.autocast("cuda", enabled=cfg.USE_AMP):
                    outputs = model(he, mihc)

                predictions[label][sample["sample_name"]] = {
                    key: value[0].float().cpu() for key, value in outputs.items()
                    if key in ["he_to_he", "he_to_mihc", "mihc_to_he", "mihc_to_mihc"]
                }

        del model
        torch.cuda.empty_cache()

    labels = list(predictions.keys())

    # Generate a reconstruction overview for each sample.
    for sample in samples:
        name = sample["sample_name"]

        he = sample["he"]
        mihc = sample["mihc"]

        rows = [
            ("HE → HE", "he_to_he", he),
            ("mIHC → HE", "mihc_to_he", he),
            ("HE → mIHC", "he_to_mihc", mihc),
            ("mIHC → mIHC", "mihc_to_mihc", mihc),
        ]

        fig, axes = plt.subplots(4, len(labels) + 1, figsize=(3 * (len(labels) + 1), 11))

        for row, (row_name, key, target) in enumerate(rows):
            axes[row, 0].imshow(to_rgb(target))
            axes[row, 0].set_title(f"{row_name}\nTarget")
            axes[row, 0].axis("off")

            for col, label in enumerate(labels, start=1):
                pred = predictions[label][name][key]

                axes[row, col].imshow(to_rgb(pred))
                axes[row, col].set_title(label)
                axes[row, col].axis("off")

        plt.suptitle(name)
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f"{name}_overview.png"), dpi=180)
        plt.close()

        # Display the three HE→mIHC channels separately.
        fig, axes = plt.subplots(3, len(labels) + 1, figsize=(3 * (len(labels) + 1), 8))

        for channel in range(3):
            axes[channel, 0].imshow(mihc[channel].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[channel, 0].set_title(f"Channel {channel}\nTarget")
            axes[channel, 0].axis("off")

            for col, label in enumerate(labels, start=1):
                pred = predictions[label][name]["he_to_mihc"]

                axes[channel, col].imshow(pred[channel].numpy(), cmap="gray", vmin=0, vmax=1)

                axes[channel, col].set_title(label)
                axes[channel, col].axis("off")

        plt.suptitle(f"{name} | HE → mIHC channels")
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f"{name}_he_to_mihc_channels.png"), dpi=180)
        plt.close()

    print("=" * 70)
    print(f"Saved {N_SAMPLES} validation samples to:")
    print(save_dir)


if __name__ == "__main__":
    main()