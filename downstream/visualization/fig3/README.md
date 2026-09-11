# Figure 3 visualization

This directory contains the plotting and source-data export scripts used for Figure 3 of the CUBE manuscript. The scientific experiment definitions are frozen; this documentation describes the repository-relative execution layout.

## Panels

- **Figure 3a** — pseudo-ST quantitative summary
- **Figure 3b** — representative pseudo-ST spatial reconstructions
- **Figure 3c** — Visium HD contiguous spatial holdout design
- **Figure 3d** — measured-ST benchmark across DeepSpot and CUBE variants
- **Figure 3e** — paired per-gene comparisons
- **Figure 3f** — whole-slide qualitative expression maps

## Paths

`fig3_paths_v3.py` resolves the repository root from its own file location. The main inputs are expected under repository-root-relative `data/`, `models/`, and `result/` trees. Figure outputs are written to:

```text
result/visualization/fig3/result/
├── source_data/
└── cache/
```

Edit `fig3_paths_v3.py` only if the released data/checkpoint archive is restored to a different repository-relative layout.

## Run

The recommended convention is to run from the repository root. To generate the default Figure 3 workflow:

```bash
python downstream/visualization/fig3/run_fig3_v3.py
```

The runner executes Figure 3a–e, exports the CUBE whole-slide prediction modes, and renders the compact Figure 3f panel. DeepSpot whole-slide inference is intentionally excluded from the default runner because it is substantially slower than plotting cached CUBE outputs.

Individual panels can also be run from the repository root, for example:

```bash
python downstream/visualization/fig3/fig3a_pseudost_summary_v3.py
python downstream/visualization/fig3/fig3c_visium_hd_split_v3.py
python downstream/visualization/fig3/fig3d_real_st_benchmark_v3.py
python downstream/visualization/fig3/fig3e_gene_level_comparison_v3.py
```

To include DeepSpot in the full whole-slide Figure 3f comparison, first generate its whole-slide cache and then render the full panel:

```bash
python downstream/visualization/fig3/fig3f_export_deepspot_whole_slide_v3.py
python downstream/visualization/fig3/fig3f_whole_slide_maps_v3.py --mode all
```

## Figure 3f methods

All decoder-adapted CUBE variants keep the pretrained H&E→UR2 encoder frozen and update only the UR2→ST decoder. The compared methods are:

- DeepSpot
- CUBE zero-shot
- CUBE pretrained-decoder fine-tuning
- CUBE scratch-decoder fine-tuning
- CUBE reset-head fine-tuning

The compact whole-slide panel uses:

```text
Measured ST | CUBE zero-shot | CUBE scratch-decoder FT | CUBE reset-head FT
```

## Source-data notes

Figure 3a writes its quantitative source table to `result/visualization/fig3/result/source_data/`. Figure 3b stores exact selected sample indices and names in the corresponding source-data CSV rather than printing long identifiers in the panel. Whole-slide maps are qualitative visualizations; the reported real-ST quantitative metrics are calculated on the fixed held-out spatial test region.
