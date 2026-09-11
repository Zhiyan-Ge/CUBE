import os, pickle
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from . import config as cfg
from .he_encoder import HEEncoder

CHECKPOINT = os.path.join(cfg.OUTPUT_DIR, "best.pt")
BATCH_SIZE = 64
SAVE_DTYPE = np.float32

class HEDataset(Dataset):
    def __init__(self, pkl_dir):
        self.paths = sorted(os.path.join(pkl_dir,x) for x in os.listdir(pkl_dir) if x.endswith(".pkl"))
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        with open(self.paths[i],"rb") as f: d=pickle.load(f)
        return {"he":torch.from_numpy(d["he_matrix_256"]).float(),"sample_name":d["sample_name"]}

def load_encoder(device):
    ckpt=torch.load(CHECKPOINT,map_location="cpu")
    state=ckpt["model"] if "model" in ckpt else ckpt
    prefixes=["model3.encoder.","module.model3.encoder."]
    enc_state={}
    for prefix in prefixes:
        enc_state={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        if enc_state: break
    if not enc_state: raise RuntimeError("No encoder weights found in checkpoint.")
    model=HEEncoder(cfg.GROUPS)
    model.load_state_dict(enc_state,strict=True)
    model.to(device).eval()
    print(f"Loaded HE encoder: {CHECKPOINT}")
    print(f"Checkpoint epoch: {ckpt.get('epoch','unknown')}")
    return model

def export(model,device,split,pkl_dir):
    ds=HEDataset(pkl_dir)
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=cfg.NUM_WORKERS,
                      pin_memory=True,persistent_workers=cfg.NUM_WORKERS>0)
    out=os.path.join(cfg.OUTPUT_DIR,"ur2_he",split)
    os.makedirs(out,exist_ok=True)
    with torch.inference_mode():
        for batch in tqdm(loader,desc=f"Export UR2 {split}",dynamic_ncols=True):
            he=batch["he"].to(device,non_blocking=True)
            ur2=model(he).cpu().numpy().astype(SAVE_DTYPE,copy=False)
            for name,x in zip(batch["sample_name"],ur2):
                np.save(os.path.join(out,f"{name}.npy"),x)
    print(f"{split}: {len(ds)} samples -> {out}")

def main():
    device=torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark=True
    model=load_encoder(device)
    export(model,device,"test",cfg.TEST_DIR)
    print("UR2 test export finished.")

if __name__=="__main__":
    main()