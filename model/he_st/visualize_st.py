import csv, os, random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as cfg
from .evaluate_st import build_loader, eval_dir, load_model, pearson_axis


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def choose_genes(rows, k):
    if not rows:
        g = list(range(min(k, 256)))
        return g, {x: "fallback" for x in g}

    def val(r, key, default=-1e9):
        try:
            return float(r[key])
        except:
            return default

    spatial = [val(x, "spatial_median", np.nan) for x in rows]
    med = np.nanmedian([x for x in spatial if np.isfinite(x)])
    high = sorted(rows, key=lambda r: val(r, "target_std"), reverse=True)
    best = sorted(rows, key=lambda r: val(r, "spatial_median"), reverse=True)
    mid = sorted(rows, key=lambda r: abs(val(r, "spatial_median", 0) - med))
    worst = list(reversed(best))
    pools = [("high variance", high), ("best spatial", best),
             ("median spatial", mid), ("worst spatial", worst)]
    chosen = []
    reasons = {}
    q = [k // 4] * 4
    for i in range(k % 4):
        q[i] += 1
    for (reason, pool), need in zip(pools, q):
        added = 0
        for r in pool:
            g = int(r["gene_index"])
            if g not in chosen:
                chosen.append(g)
                reasons[g] = reason
                added += 1
            if added >= need:
                break
    for r in best:
        g = int(r["gene_index"])
        if len(chosen) >= k:
            break
        if g not in chosen:
            chosen.append(g)
            reasons[g] = "fill"
    return chosen[:k], reasons


def main():
    path = cfg.EVAL_CHECKPOINT
    split = cfg.EVAL_SPLIT
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(path, device)
    _, ds = build_loader(split)
    out = eval_dir(path, split)
    vis = os.path.join(out, "st_visualizations")
    os.makedirs(vis, exist_ok=True)

    rows = read_rows(os.path.join(out, "gene_spatial_diagnostics_he_to_st.csv"))
    genes, reasons = choose_genes(rows, cfg.EVAL_NUM_GENES)
    global_r = {int(r["gene_index"]): float(r["global_gene_pearson"]) for r in rows} if rows else {}
    rng = random.Random(cfg.SEED)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:min(cfg.EVAL_NUM_SAMPLES, len(ds))]
    print(f"checkpoint={path} | epoch={epoch} | split={split} | "
          f"genes={[(g, reasons[g]) for g in genes]}")

    with torch.inference_mode():
        for i in idx:
            s = ds[i]
            he = s["he"].unsqueeze(0).to(device)
            u = model.model3.encoder(he)
            pred = model.model3.st_decoder(u)[0].cpu().numpy()
            target = s["st"].numpy()
            img = s["he"].numpy().transpose(1, 2, 0)
            fig, ax = plt.subplots(len(genes) + 1, 3, figsize=(9, 2.5 * (len(genes) + 1)))
            ax[0, 0].imshow(img)
            ax[0, 0].set_title(f"H&E\n{s['sample_name']}")
            [ax[0, j].axis("off") for j in range(3)]
            for r, g in enumerate(genes, 1):
                t = target[:, :, g]
                p = pred[:, :, g]
                vmin = min(t.min(), p.min())
                vmax = max(t.max(), p.max())
                sr = float(pearson_axis(p.reshape(1, -1), t.reshape(1, -1), 1)[0])
                res = p - t
                lim = max(abs(res.min()), abs(res.max()), 1e-6)
                ax[r, 0].imshow(t, vmin=vmin, vmax=vmax, cmap="viridis")
                ax[r, 0].set_title(f"Gene {g} Target\n{reasons[g]}")
                ax[r, 1].imshow(p, vmin=vmin, vmax=vmax, cmap="viridis")
                ax[r, 1].set_title(f"Pred | spatial r={sr:.3f}\n"
                                   f"global r={global_r.get(g, float('nan')):.3f}")
                ax[r, 2].imshow(res, vmin=-lim, vmax=lim, cmap="coolwarm")
                ax[r, 2].set_title("Residual: Pred-Target")
                [ax[r, j].axis("off") for j in range(3)]
            plt.tight_layout()
            name = str(s["sample_name"]).replace("/", "_")
            plt.savefig(os.path.join(vis, f"sample_{name}.png"), dpi=180)
            plt.close()
    print(f"Saved: {vis}")


if __name__ == "__main__":
    main()