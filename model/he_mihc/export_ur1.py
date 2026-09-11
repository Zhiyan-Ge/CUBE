import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from . import config as cfg
from .dataset import HEMIHCDataset
from .encoder import ImageEncoder

CHECKPOINT = os.path.join(cfg.OUTPUT_DIR, "best_pearson.pt")
BATCH_SIZE = 32          # encoder-only
SAVE_DTYPE = np.float32  

def load_encoder(device):
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    prefix = "model1.encoder."
    enc_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not enc_state:
        prefix = "module.model1.encoder."
        enc_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not enc_state:
        raise RuntimeError("No encoder state found in checkpoint.")
    model = ImageEncoder(cfg.GROUPS)
    model.load_state_dict(enc_state, strict=True)
    model.to(device).eval()
    print(f"Loaded HE encoder from: {CHECKPOINT}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    return model

def export_split(model, device, split, pkl_dir):
    dataset = HEMIHCDataset(pkl_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True,
                        persistent_workers=cfg.NUM_WORKERS > 0)
    out_dir = os.path.join(cfg.OUTPUT_DIR, "ur1_he", split)
    os.makedirs(out_dir, exist_ok=True)
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Export UR1 {split}", dynamic_ncols=True):
            he = batch["he"].to(device, non_blocking=True)
            ur1 = model(he).cpu().numpy().astype(SAVE_DTYPE, copy=False)
            for name, feat in zip(batch["sample_name"], ur1):
                np.save(os.path.join(out_dir, f"{name}.npy"), feat)
    print(f"{split}: {len(dataset)} samples -> {out_dir}")

def main():
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = load_encoder(device)
    export_split(model, device, "train", cfg.TRAIN_DIR)
    export_split(model, device, "val", cfg.VAL_DIR)
    export_split(model, device, "test", cfg.TEST_DIR)
    print("UR1 export finished.")

if __name__ == "__main__":
    main()