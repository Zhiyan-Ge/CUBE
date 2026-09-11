import os, json, pickle, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from . import config as cfg
from .dataset import URDataset
from .model import URModel

NAMES = ["DAPI", "CD3", "panCK"]
EPS = 1e-12

def get_dirs(split):
    if split == "val":
        return cfg.VAL_PKL_DIR, cfg.VAL_UR1_DIR, cfg.VAL_UR2_DIR
    if split == "test":
        return cfg.TEST_PKL_DIR, cfg.TEST_UR1_DIR, cfg.TEST_UR2_DIR
    raise ValueError(f"EVAL_SPLIT must be val/test, got {split}")

def load_model(device):
    ckpt = torch.load(cfg.EVAL_CHECKPOINT, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = {k[7:] if k.startswith("module.") else k:v for k,v in state.items()}
    model = URModel(cfg.DIM, cfg.HEADS, cfg.FFN_DIM, cfg.DROPOUT).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt.get("epoch", -1)

def train_concept_mean():
    values = []
    for f in sorted(os.listdir(cfg.TRAIN_PKL_DIR)):
        if not f.endswith(".pkl"): continue
        with open(os.path.join(cfg.TRAIN_PKL_DIR, f), "rb") as h:
            values.append(np.asarray(pickle.load(h)["concept_scores"], dtype=np.float64))
    return np.stack(values).mean(0)

def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    x, y = x-x.mean(), y-y.mean()
    den = np.sqrt(np.sum(x*x)*np.sum(y*y))
    return float(np.sum(x*y)/den) if den > EPS else None

def metrics(pred, target, baseline):
    mse = float(np.mean((pred-target)**2))
    mae = float(np.mean(np.abs(pred-target)))
    base_mse = float(np.mean((baseline-target)**2))
    ss_res = float(np.sum((pred-target)**2))
    ss_tot = float(np.sum((target-target.mean())**2))
    return {
        "mse": mse, "mae": mae, "rmse": float(np.sqrt(mse)),
        "pearson": pearson(pred, target),
        "r2": 1.0-ss_res/ss_tot if ss_tot > EPS else None,
        "target_mean": float(target.mean()), "target_std": float(target.std()),
        "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
        "pred_std_over_target_std": float(pred.std()/target.std()) if target.std() > EPS else None,
        "train_mean_baseline": float(baseline),
        "baseline_mse": base_mse,
        "mse_over_baseline": mse/base_mse if base_mse > EPS else None,
        "baseline_improvement_percent": 100.0*(1.0-mse/base_mse) if base_mse > EPS else None
    }

def main():
    split = cfg.EVAL_SPLIT
    pkl_dir, ur1_dir, ur2_dir = get_dirs(split)
    ds = URDataset(pkl_dir, ur1_dir, ur2_dir)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True)
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(device)

    preds, targets, names = [], [], []
    with torch.inference_mode():
        for b in loader:
            ur1 = b["ur1"].to(device, non_blocking=True)
            ur2 = b["ur2"].to(device, non_blocking=True)
            coord = b["coord"].to(device, non_blocking=True)
            preds.append(model(ur1, ur2, coord).cpu().numpy())
            targets.append(b["concept"].numpy())
            names.extend(b["sample_name"])

    pred = np.concatenate(preds).astype(np.float64)
    target = np.concatenate(targets).astype(np.float64)
    train_mean = train_concept_mean()
    baseline = np.broadcast_to(train_mean, target.shape)

    overall_mse = float(np.mean((pred-target)**2))
    overall_mae = float(np.mean(np.abs(pred-target)))
    baseline_mse = float(np.mean((baseline-target)**2))
    per_concept = {NAMES[i]: metrics(pred[:,i], target[:,i], train_mean[i]) for i in range(3)}

    report = {
        "checkpoint": cfg.EVAL_CHECKPOINT, "checkpoint_epoch": epoch,
        "split": split, "samples": len(ds),
        "overall": {
            "mse": overall_mse, "mae": overall_mae, "rmse": float(np.sqrt(overall_mse)),
            "train_mean_baseline_mse": baseline_mse,
            "mse_over_baseline": overall_mse/baseline_mse,
            "baseline_improvement_percent": 100.0*(1.0-overall_mse/baseline_mse)
        },
        "train_concept_mean": train_mean.tolist(),
        "per_concept": per_concept
    }

    tag = os.path.splitext(os.path.basename(cfg.EVAL_CHECKPOINT))[0]
    out = os.path.join(cfg.OUTPUT_DIR, "evaluation", f"{tag}_{split}")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, "ur_evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(out, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_name"]+[f"target_{x}" for x in NAMES]+[f"pred_{x}" for x in NAMES])
        for n,t,p in zip(names,target,pred): w.writerow([n,*t,*p])

    print("="*78)
    print(f"checkpoint={cfg.EVAL_CHECKPOINT} | epoch={epoch} | split={split} | N={len(ds)}")
    print(f"Overall MSE={overall_mse:.6f} | MAE={overall_mae:.6f} | RMSE={np.sqrt(overall_mse):.6f}")
    print(f"Mean baseline MSE={baseline_mse:.6f} | ratio={overall_mse/baseline_mse:.3f} | improvement={100*(1-overall_mse/baseline_mse):.1f}%")
    print("-"*78)
    for name in NAMES:
        m = per_concept[name]
        print(f"{name:5s} | MSE={m['mse']:.6f} MAE={m['mae']:.6f} "
              f"r={m['pearson']:.4f} R2={m['r2']:.4f} "
              f"std={m['pred_std']:.4f}/{m['target_std']:.4f} "
              f"baseline={m['baseline_mse']:.6f} improve={m['baseline_improvement_percent']:.1f}%")
    print("-"*78)
    print(f"Saved: {out}")
    print("="*78)

if __name__ == "__main__":
    main()