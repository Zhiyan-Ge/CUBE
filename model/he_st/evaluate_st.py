import json, os, time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as cfg
from .dataset import HESTDataset
from .model import HESTBranch


EPS = 1e-8


def split_dir(split):
    m = {"train": cfg.TRAIN_DIR, "val": cfg.VAL_DIR, "test": getattr(cfg, "TEST_DIR", None)}
    if split not in m or not m[split]:
        raise ValueError(f"Invalid EVAL_SPLIT: {split}")
    return m[split]


def build_loader(split):
    ds = HESTDataset(split_dir(split), cfg.ST_STATS_PATH)
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                      pin_memory=True, persistent_workers=cfg.NUM_WORKERS > 0), ds


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    state = ck.get("model", ck.get("state_dict", ck))
    model = HESTBranch(cfg.GROUPS).to(device)
    model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state.items()}, strict=True)
    model.eval()
    return model, int(ck.get("epoch", -1)) if isinstance(ck, dict) else -1


def pearson_axis(a, b, axis=-1):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ac = a - a.mean(axis=axis, keepdims=True)
    bc = b - b.mean(axis=axis, keepdims=True)
    va = (ac * ac).sum(axis=axis)
    vb = (bc * bc).sum(axis=axis)
    den = np.sqrt(va * vb)
    num = (ac * bc).sum(axis=axis)
    out = np.full(np.shape(den), np.nan, np.float64)
    ok = (va > EPS) & (vb > EPS) & np.isfinite(den)
    out[ok] = num[ok] / den[ok]
    return np.clip(out, -1, 1)


def summary(x, thresholds=()):
    x = np.asarray(x, np.float64)
    v = x[np.isfinite(x)]
    d = {"valid": int(v.size), "invalid": int(x.size - v.size)}
    if not v.size:
        return d | {"mean": None, "median": None, "std": None, "p25": None, "p75": None,
                    "min": None, "max": None}
    d |= {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()),
          "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
          "min": float(v.min()), "max": float(v.max())}
    for t in thresholds:
        k = str(t).replace(".", "_")
        d[f"count_gt_{k}"] = int((v > t).sum())
        d[f"frac_gt_{k}"] = float((v > t).mean())
    return d


def decompose(pred, target):
    n, h, w, g = pred.shape
    ps = pred.transpose(0, 3, 1, 2).reshape(n, g, h * w)
    ts = target.transpose(0, 3, 1, 2).reshape(n, g, h * w)
    spatial = pearson_axis(ps, ts, 2)
    pm = pred.mean((1, 2))
    tm = target.mean((1, 2))
    abundance = pearson_axis(pm.T, tm.T, 1)
    global_gene = pearson_axis(ps.transpose(1, 0, 2).reshape(g, -1),
                               ts.transpose(1, 0, 2).reshape(g, -1), 1)
    return spatial, abundance, global_gene


def basic_metrics(pred, target):
    mse = float(np.mean((pred - target) ** 2))
    zero = float(np.mean(target ** 2))
    sample = pearson_axis(pred.reshape(len(pred), -1), target.reshape(len(target), -1), 1)
    _, _, gene = decompose(pred, target)
    return {"mse": mse, "zero_baseline_mse": zero,
            "mse_over_zero_baseline": mse / zero if zero > EPS else None,
            "global_pearson": float(pearson_axis(pred.reshape(1, -1), target.reshape(1, -1), 1)[0]),
            "sample_wise_pearson": summary(sample), "gene_wise_global_pearson": summary(gene)}


def eval_dir(path, split):
    return os.path.join(cfg.OUTPUT_DIR, "evaluation", f"{os.path.splitext(os.path.basename(path))[0]}_{split}")


def collect(model, loader, device, need_st_st=True):
    he_st = []
    st_st = []
    targets = []
    names = []
    with torch.inference_mode():
        for b in loader:
            he = b["he"].to(device, non_blocking=True)
            st = b["st"].to(device, non_blocking=True)
            u = model.model3.encoder(he)
            he_st.append(model.model3.st_decoder(u).cpu())
            targets.append(st.cpu())
            names.extend(list(b["sample_name"]))
            if need_st_st:
                us = model.model4.encoder(st)
                st_st.append(model.model4.st_decoder(us).cpu())
    pred = torch.cat(he_st).numpy()
    target = torch.cat(targets).numpy()
    st_pred = torch.cat(st_st).numpy() if need_st_st else None
    return pred, st_pred, target, names


def main():
    path = cfg.EVAL_CHECKPOINT
    split = cfg.EVAL_SPLIT
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(path, device)
    loader, _ = build_loader(split)
    t0 = time.time()
    pred, st_pred, target, _ = collect(model, loader, device)
    spatial, abundance, gene = decompose(pred, target)
    he = basic_metrics(pred, target)
    st = basic_metrics(st_pred, target)

    he["target_global_mean"] = float(target.mean())
    he["target_global_std"] = float(target.std())
    he["prediction_global_mean"] = float(pred.mean())
    he["prediction_global_std"] = float(pred.std())
    he["pred_std_over_target_std"] = float(pred.std() / target.std())
    he["within_patch_spatial_pearson"] = summary(spatial, (0, 0.1, 0.3, 0.5, 0.7))
    he["cross_sample_abundance_pearson"] = summary(abundance, (0, 0.3, 0.5, 0.7, 0.8))
    he["gene_wise_global_pearson"] = summary(gene, (0, 0.3, 0.5, 0.7, 0.8))

    report = {"checkpoint": path, "checkpoint_epoch": epoch, "split": split, "num_samples": len(target),
              "HE->ST": he, "ST->ST": st, "evaluation_seconds": time.time() - t0}
    out = eval_dir(path, split)
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, "st_evaluation.json")
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"checkpoint={path} | epoch={epoch} | split={split} | N={len(target)}")
    print(f"HE->ST MSE={he['mse']:.4f} | zero={he['zero_baseline_mse']:.4f} | "
          f"ratio={he['mse_over_zero_baseline']:.3f} | global r={he['global_pearson']:.4f}")
    print(f"gene-global r: mean={he['gene_wise_global_pearson']['mean']:.4f} "
          f"median={he['gene_wise_global_pearson']['median']:.4f}")
    print(f"abundance r:   mean={he['cross_sample_abundance_pearson']['mean']:.4f} "
          f"median={he['cross_sample_abundance_pearson']['median']:.4f}")
    print(f"spatial r:     mean={he['within_patch_spatial_pearson']['mean']:.4f} "
          f"median={he['within_patch_spatial_pearson']['median']:.4f} "
          f">0.3={he['within_patch_spatial_pearson']['frac_gt_0_3']:.3f} "
          f">0.5={he['within_patch_spatial_pearson']['frac_gt_0_5']:.3f}")
    print(f"ST->ST MSE={st['mse']:.4f} | global r={st['global_pearson']:.4f}\nSaved: {fn}")


if __name__ == "__main__":
    main()