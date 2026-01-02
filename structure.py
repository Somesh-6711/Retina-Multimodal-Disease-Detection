# structure.py
from pathlib import Path

def create_structure(root: str = "."):
    root = Path(root).resolve()
    print(f"Repo root: {root}")

    dirs = [
        "data/raw/fundus",
        "data/raw/oct",
        "data/processed/fundus",
        "data/processed/oct",
        "src/configs",
        "src/data",
        "src/models",
        "src/training",
        "src/inference",
        "src/explainability",
        "src/utils",
        "notebooks",
        "outputs/fundus",
        "outputs/oct",
        "outputs/fusion",
    ]

    for d in dirs:
        p = root / d
        p.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {p}")

    # create empty __init__.py files for src package
    for pkg in ["src", "src/data", "src/models", "src/training", "src/inference", "src/explainability", "src/utils", "src/configs"]:
        p = root / pkg / "__init__.py"
        p.touch(exist_ok=True)
        print(f"[INIT] {p}")

if __name__ == "__main__":
    create_structure()
