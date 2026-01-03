import os
from pathlib import Path
import glob

import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# Paths (adjust if your layout differs)
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

FUNDUS_RAW_DIR = REPO_ROOT / "data" / "raw" / "fundus" / "aptos2019" / "train_images"
FUNDUS_GRADCAM_DIR = REPO_ROOT / "outputs" / "fundus" / "gradcam"

OCT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "oct" / "kermany2018"
OCT_GRADCAM_DIR = REPO_ROOT / "outputs" / "oct" / "gradcam"

OUT_FIG_DIR = REPO_ROOT / "outputs"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_raw_fundus_from_gradcam(gradcam_filename: str) -> Path:
    """
    gradcam_filename example:
        'e1fb532f55df_true3_pred4_gradcam.png'

    Original fundus image is typically:
        data/raw/fundus/aptos2019/train_images/e1fb532f55df.png
    """
    stem = gradcam_filename.split("_true")[0]
    candidates = list(FUNDUS_RAW_DIR.glob(f"{stem}.*"))
    if not candidates:
        raise FileNotFoundError(f"No raw fundus image found for id={stem}")
    return candidates[0]


def find_raw_oct_from_gradcam(gradcam_filename: str) -> Path:
    """
    gradcam_filename example:
        'DME-9583225-1_trueDME_predDME_gradcam.png'

    Original OCT file is usually:
        data/raw/oct/kermany2018/**/DME-9583225-1.*
    We search recursively under OCT_RAW_ROOT.
    """
    stem = gradcam_filename.split("_true")[0]  # 'DME-9583225-1'
    pattern = str(OCT_RAW_ROOT / "**" / f"{stem}.*")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No raw OCT image found for id={stem}")
    return Path(candidates[0])


def make_before_after_figure(raw_path: Path, gradcam_path: Path,
                             title_left: str, title_right: str,
                             out_path: Path):
    """Create a side-by-side (before vs after) panel."""
    raw_img = Image.open(raw_path).convert("RGB")
    gc_img = Image.open(gradcam_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(raw_img)
    axes[0].set_title(title_left)
    axes[1].imshow(gc_img)
    axes[1].set_title(title_right)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[SAVE] {out_path}")


def main():
    # -----------------------------
    # Fundus before/after
    # -----------------------------
    fundus_gc_name = "e1fb532f55df_true3_pred4_gradcam.png"  # choose your favourite example
    fundus_gc_path = FUNDUS_GRADCAM_DIR / fundus_gc_name
    fundus_raw_path = find_raw_fundus_from_gradcam(fundus_gc_name)

    make_before_after_figure(
        raw_path=fundus_raw_path,
        gradcam_path=fundus_gc_path,
        title_left="Fundus – Raw",
        title_right="Fundus – Grad-CAM",
        out_path=OUT_FIG_DIR / "fundus_before_after.png",
    )

    # -----------------------------
    # OCT before/after
    # -----------------------------
    oct_gc_name = "DME-9583225-1_trueDME_predDME_gradcam.png"  # example from your Grad-CAMs
    oct_gc_path = OCT_GRADCAM_DIR / oct_gc_name
    oct_raw_path = find_raw_oct_from_gradcam(oct_gc_name)

    make_before_after_figure(
        raw_path=oct_raw_path,
        gradcam_path=oct_gc_path,
        title_left="OCT – Raw",
        title_right="OCT – Grad-CAM",
        out_path=OUT_FIG_DIR / "oct_before_after.png",
    )


if __name__ == "__main__":
    main()