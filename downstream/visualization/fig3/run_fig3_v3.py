import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SCRIPTS = [
    "fig3a_pseudost_summary_v3.py",
    "fig3b_export_pseudost_examples_v3.py",
    "fig3b_plot_pseudost_examples_v3.py",
    "fig3c_visium_hd_split_v3.py",
    "fig3d_real_st_benchmark_v3.py",
    "fig3e_gene_level_comparison_v3.py",
    "fig3f_export_cube_whole_slide_v3.py",
]


def run_script(name, *args):
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / name), *args],
        check=True,
        cwd=PROJECT_ROOT,
    )


def main():
    for script in SCRIPTS:
        print(f"\n{'=' * 80}\nRunning {script}\n{'=' * 80}")
        run_script(script)
    run_script("fig3f_whole_slide_maps_v3.py", "--mode", "compact")
    print("\nCompact Fig3 draft complete.")
    print("For the six-column Fig3F, run fig3f_export_deepspot_whole_slide_v3.py once, then plot with --mode all.")


if __name__ == "__main__":
    main()
