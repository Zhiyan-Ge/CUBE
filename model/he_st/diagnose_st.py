import csv, json, os, time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as cfg
from .evaluate_st import EPS, build_loader, collect, decompose, eval_dir, load_model, summary


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def hist(x, path, title):
    v = np.asarray(x)
    v = v[np.isfinite(v)]
    plt.figure(figsize=(7, 5))
    plt.hist(v, bins=30, range=(-1, 1))
    plt.xlabel("Pearson r")
    plt.ylabel("Count")
    plt.title(f"{title}\nvalid={len(v)}, mean={v.mean():.3f}, median={np.median(v):.3f}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    path = cfg.EVAL_CHECKPOINT
    split = cfg.EVAL_SPLIT
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(path, device)
    loader, _ = build_loader(split)
    t0 = time.time()
    pred, _, target, names = collect(model, loader, device, False)
    spatial, abundance, global_gene = decompose(pred, target)

    n, h, w, g = pred.shape
    ps = pred.transpose(0, 3, 1, 2).reshape(n, g, -1)
    ts = target.transpose(0, 3, 1, 2).reshape(n, g, -1)
    gene_rows = []
    for j in range(g):
        p = ps[:, j].reshape(-1)
        t = ts[:, j].reshape(-1)
        tr = float(t.std())
        pr = float(p.std())
        s = spatial[:, j]
        v = s[np.isfinite(s)]
        gene_rows.append({"gene_index": j, "target_std": tr, "pred_std": pr,
                          "std_ratio": pr / tr if tr > EPS else np.nan,
                          "mse": float(np.mean((p - t) ** 2)),
                          "global_gene_pearson": global_gene[j], "abundance_pearson": abundance[j],
                          "spatial_valid": len(v), "spatial_mean": float(v.mean()) if len(v) else np.nan,
                          "spatial_median": float(np.median(v)) if len(v) else np.nan,
                          "spatial_p25": float(np.percentile(v, 25)) if len(v) else np.nan,
                          "spatial_p75": float(np.percentile(v, 75)) if len(v) else np.nan})

    sample_rows = []
    for i, name in enumerate(names):
        v = spatial[i][np.isfinite(spatial[i])]
        sample_rows.append({"sample_name": name, "valid_genes": len(v),
                            "spatial_mean": float(v.mean()) if len(v) else np.nan,
                            "spatial_median": float(np.median(v)) if len(v) else np.nan,
                            "spatial_p25": float(np.percentile(v, 25)) if len(v) else np.nan,
                            "spatial_p75": float(np.percentile(v, 75)) if len(v) else np.nan,
                            "genes_r_gt_0": int((v > 0).sum()), "genes_r_gt_0_3": int((v > 0.3).sum()),
                            "genes_r_gt_0_5": int((v > 0.5).sum())})

    out = eval_dir(path, split)
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "gene_spatial_diagnostics_he_to_st.csv"), gene_rows)
    write_csv(os.path.join(out, "sample_spatial_diagnostics_he_to_st.csv"), sample_rows)

    x = np.array([r["target_std"] for r in gene_rows])
    y = np.array([r["pred_std"] for r in gene_rows])
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=14)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    plt.plot([lo, hi], [lo, hi], "--", label="y=x")
    plt.xlabel("Target std")
    plt.ylabel("Pred std")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "pred_std_vs_target_std.png"), dpi=180)
    plt.close()

    hist(global_gene, os.path.join(out, "gene_global_pearson_hist.png"), "Global gene Pearson (N×16×16)")
    hist(abundance, os.path.join(out, "gene_abundance_pearson_hist.png"),
         "Cross-sample gene abundance Pearson")
    hist(spatial.reshape(-1), os.path.join(out, "within_patch_spatial_pearson_hist.png"),
         "Within-patch spatial Pearson")

    report = {"checkpoint": path, "checkpoint_epoch": epoch, "split": split,
              "global_gene": summary(global_gene), "abundance": summary(abundance),
              "within_patch_spatial": summary(spatial, (0, 0.3, 0.5, 0.7)),
              "std_ratio": summary([r["std_ratio"] for r in gene_rows]), "seconds": time.time() - t0}
    with open(os.path.join(out, "spatial_diagnostics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"checkpoint={path} | epoch={epoch} | split={split}")
    print(f"global gene r mean/median={report['global_gene']['mean']:.4f}/"
          f"{report['global_gene']['median']:.4f}")
    print(f"abundance r  mean/median={report['abundance']['mean']:.4f}/"
          f"{report['abundance']['median']:.4f}")
    print(f"spatial r    mean/median={report['within_patch_spatial']['mean']:.4f}/"
          f"{report['within_patch_spatial']['median']:.4f} | "
          f">0.3={report['within_patch_spatial']['frac_gt_0_3']:.3f} | "
          f">0.5={report['within_patch_spatial']['frac_gt_0_5']:.3f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()