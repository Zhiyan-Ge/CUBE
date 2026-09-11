import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SCRIPTS = [
    "fig4a_validation_complementarity_v1.py",
    "fig4b_test_pearson_tissue_threshold_v1.py",
    "fig4c_mse_advantage_tissue_threshold_v1.py",
]


def main():
    for script in SCRIPTS:
        print("\n" + "=" * 80)
        print(f"Running {script}")
        print("=" * 80)
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / script)],
            check=True,
            cwd=PROJECT_ROOT,
        )


if __name__ == "__main__":
    main()
