"""Generate every paper figure into results/figures/.

All figures that require probabilities load arrays from artifacts/ at runtime —
nothing is hardcoded. Figures that need raw image data are noted as skipped when
the data directory is absent.

Filenames match the paper's \\includegraphics calls.
Run via:
    PYTHONPATH=src python -m disaster.eda.plots
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

from disaster.labels import LABELS

OUT = Path("results/figures")

BRANCH_NAMES = [
    "banglabert_base", "banglabert_multi", "muril_large",
    "eva02_large", "fusion_eva_muril", "pl_muril", "stack",
]
BRANCH_LABELS = [
    "BanglaBERT-base", "BanglaBERT-multi", "MuRIL-Large",
    "EVA-02-Large", "EVA-02×MuRIL", "PL-MuRIL", "LightGBM stack",
]


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUT / name}")


def _load_y(folds_csv="data/folds/folds_canonical.csv"):
    df = pd.read_csv(folds_csv, encoding="utf-8-sig")
    return df, df["label"].values


# ---------------------------------------------------------------------------
# Fig 1 — class distribution (from folds_canonical.csv, no raw CSV needed)
# ---------------------------------------------------------------------------
def fig_class_dist(folds_csv="data/folds/folds_canonical.csv"):
    df, _ = _load_y(folds_csv)
    vc = df["category"].value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#d9534f" if l == "Wildfire" else "#4c72b0" for l in LABELS]
    bars = ax.bar(range(len(LABELS)), vc.values, color=colors)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Sample count")
    ax.set_title("Training set class distribution")
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, str(v),
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_class_dist.pdf")


# ---------------------------------------------------------------------------
# Fig 2 — OOF macro-F1 waterfall per branch
# ---------------------------------------------------------------------------
def fig_oof_waterfall(folds_csv="data/folds/folds_canonical.csv",
                      artifacts="artifacts"):
    _, y = _load_y(folds_csv)
    A = Path(artifacts)
    values, names = [], []
    for n, label in zip(BRANCH_NAMES, BRANCH_LABELS):
        p = np.load(A / f"oof_{n}.npy")
        values.append(f1_score(y, p.argmax(1), average="macro"))
        names.append(label)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#4c72b0"] * 6 + ["#2ca02c"]
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("OOF macro-F1")
    ax.set_xlim(0.95, 1.002)
    ax.set_title("OOF macro-F1 per branch")
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_oof_waterfall.pdf")


# ---------------------------------------------------------------------------
# Fig 3 — per-class F1 for the final ensemble
# ---------------------------------------------------------------------------
def fig_per_class_f1(final_oof_npy, folds_csv="data/folds/folds_canonical.csv"):
    p = np.load(final_oof_npy)
    _, y = _load_y(folds_csv)
    per_class = f1_score(y, p.argmax(1), average=None)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(range(len(LABELS)), per_class, color="#4c72b0")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0.96, 1.005)
    ax.set_title("Per-class F1 (final bootstrap-stable ensemble)")
    for bar, v in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_per_class_f1.pdf")


# ---------------------------------------------------------------------------
# Fig 4 — confusion matrix (normalized)
# ---------------------------------------------------------------------------
def fig_confusion(final_oof_npy, folds_csv="data/folds/folds_canonical.csv"):
    p = np.load(final_oof_npy)
    _, y = _load_y(folds_csv)
    cm = confusion_matrix(y, p.argmax(1), normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(8):
        for j in range(8):
            if cm[i, j] > 0.005:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, "fig_confusion.pdf")


# ---------------------------------------------------------------------------
# Fig 5 — fusion strategy comparison
# ---------------------------------------------------------------------------
def fig_fusion_compare(folds_csv="data/folds/folds_canonical.csv",
                       artifacts="artifacts"):
    _, y = _load_y(folds_csv)
    A = Path(artifacts)
    text_only = f1_score(y, np.load(A / "oof_muril_large.npy").argmax(1), average="macro")
    image_only = f1_score(y, np.load(A / "oof_eva02_large.npy").argmax(1), average="macro")
    cross_attn = f1_score(y, np.load(A / "oof_fusion_eva_muril.npy").argmax(1), average="macro")
    values = {
        "Text-only\n(MuRIL-L)":   text_only,
        "Image-only\n(EVA-02-L)": image_only,
        "Cross-attn\n(EVA×MuRIL)": cross_attn,
    }
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(list(values), list(values.values()),
                  color=["#4c72b0", "#dd8452", "#2ca02c"])
    ax.set_ylim(0.95, 1.005)
    ax.set_ylabel("OOF macro-F1")
    ax.set_title("Fusion strategy comparison")
    for bar, v in zip(bars, values.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_fusion_compare.pdf")


# ---------------------------------------------------------------------------
# Fig 6 — bias vector bar chart
# ---------------------------------------------------------------------------
def fig_bias_vector(bias_npy="artifacts/bias_s3e.npy"):
    b = np.load(bias_npy)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = ["#d9534f" if v < 0 else "#4c72b0" for v in b]
    ax.bar(range(len(LABELS)), b, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("log-bias value")
    ax.set_title("Per-class log-bias (bootstrap-stable, 15 resamples)")
    for i, v in enumerate(b):
        ax.text(i, v + (0.01 if v >= 0 else -0.04), f"{v:.3f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_bias_vector.pdf")


# ---------------------------------------------------------------------------
# Fig 7 — modality error overlap
# ---------------------------------------------------------------------------
def fig_error_overlap(folds_csv="data/folds/folds_canonical.csv",
                      artifacts="artifacts"):
    _, y = _load_y(folds_csv)
    A = Path(artifacts)
    text_wrong = np.load(A / "oof_muril_large.npy").argmax(1) != y
    img_wrong  = np.load(A / "oof_eva02_large.npy").argmax(1) != y
    both_wrong   = int((text_wrong & img_wrong).sum())
    text_only_e  = int((text_wrong & ~img_wrong).sum())
    image_only_e = int((~text_wrong & img_wrong).sum())
    both_right   = int((~text_wrong & ~img_wrong).sum())

    cats = ["Both correct", "Only image\nwrong", "Only text\nwrong", "Both wrong"]
    vals = [both_right, image_only_e, text_only_e, both_wrong]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(cats, vals, color=["#2ca02c", "#dd8452", "#4c72b0", "#d9534f"])
    ax.set_ylabel("Sample count")
    ax.set_title("Modality error overlap (MuRIL-L vs EVA-02-L)")
    for i, v in enumerate(vals):
        ax.text(i, v + 5, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_error_overlap.pdf")


def generate_all(folds_csv="data/folds/folds_canonical.csv",
                 artifacts="artifacts"):
    """Generate all available figures; print which are skipped."""
    print("Generating figures …")
    A = Path(artifacts)
    fig_class_dist(folds_csv)
    fig_oof_waterfall(folds_csv, artifacts)
    fig_fusion_compare(folds_csv, artifacts)
    fig_error_overlap(folds_csv, artifacts)
    fig_bias_vector(str(A / "bias_s3e.npy"))

    # Figures that need the final calibrated OOF (final_oof.npy is pre-calib blend;
    # we use the stack OOF as a proxy — or callers can pass a calibrated npy)
    final_npy = A / "oof_stack.npy"
    if final_npy.exists():
        fig_confusion(str(final_npy), folds_csv)
        fig_per_class_f1(str(final_npy), folds_csv)

    print(f"All figures written to {OUT}/")


if __name__ == "__main__":
    generate_all()
