#!/usr/bin/env python3
"""Build fixed KEGG gene lists used by the Fig. 5D real-ST program analysis."""

from pathlib import Path

import pandas as pd

GENE_MAPPING = Path("./result/01_real_st/data/HD/visium_hd_16um_paired/gene_mapping.csv")
OUT_DIR = Path("./result/05_real_st_program/data/program_definition")

PROGRAMS = {
    "ECM_receptor_interaction": {
        "kegg_id": "hsa04512",
        "genes": ["COL1A1", "COL1A2", "COL4A1", "COL4A2", "COL6A1", "COL6A2", "COL6A3", "COMP", "FN1", "HSPG2", "ITGA11", "ITGA5", "SPP1", "THBS1", "THBS2", "TNC", "VWF"],
    },
    "Complement_and_coagulation_cascades": {
        "kegg_id": "hsa04610",
        "genes": ["A2M", "C1QB", "C1QC", "C1R", "C1S", "C3", "CFB", "CD55", "SERPING1", "SERPINE1", "PLAU", "VWF"],
    },
}


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(GENE_MAPPING).sort_values("model_channel").reset_index(drop=True)
    mapping["in_visium_hd"] = as_bool(mapping["in_visium_hd"])

    rows = []
    for program, info in PROGRAMS.items():
        order = {gene: i for i, gene in enumerate(info["genes"])}
        selected = mapping.loc[
            mapping["gene_name"].isin(info["genes"]) & mapping["in_visium_hd"],
            ["model_channel", "gene_name", "deepspot_prediction_index", "visium_hd_var_index", "in_visium_hd"],
        ].copy()
        selected["program"] = program
        selected["kegg_id"] = info["kegg_id"]
        selected["program_gene_order"] = selected["gene_name"].map(order)
        rows.append(selected.sort_values("program_gene_order"))

    gene_list = pd.concat(rows, ignore_index=True)
    columns = ["program", "kegg_id", "program_gene_order", "model_channel", "gene_name",
               "deepspot_prediction_index", "visium_hd_var_index", "in_visium_hd"]
    gene_list = gene_list[columns]
    gene_list.to_csv(OUT_DIR / "fig5d_kegg_program_gene_list.csv", index=False)

    for program in PROGRAMS:
        genes = gene_list.loc[gene_list["program"] == program, "gene_name"].tolist()
        print(f"{program}: {len(genes)} genes")
        print(", ".join(genes))
    print(f"saved: {OUT_DIR / 'fig5d_kegg_program_gene_list.csv'}")


if __name__ == "__main__":
    main()