"""Generate every paper figure into results/figures/.

Figures that depend on probabilities load arrays from artifacts/ at runtime,
so nothing is hardcoded. Filenames match the paper's \\includegraphics calls.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

from disaster.labels import LABELS

OUT = Path("results/figures")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def fig_class_dist(train_csv):
    df = pd.read_csv(train_csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    vc = df["category"].value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = ["#d9534f" if l == "Wildfire" else "#4c72b0" for l in LABELS]
    ax.bar(range(len(LABELS)), vc.values, color=colors)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_ylabel("count")
    _save(fig, "fig_class_dist.pdf")


def fig_confusion(mix_oof_npy, folds_csv):
    p = np.load(mix_oof_npy)
    y = pd.read_csv(folds_csv, encoding="utf-8-sig")["label"].values
    cm = confusion_matrix(y, p.argmax(1), normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(LABELS, rotation=45, ha="right"); ax.set_yticklabels(LABELS)
    for i in range(8):
        for j in range(8):
            if cm[i, j] > 0.01:
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if cm[i, j] > .5 else "black")
    fig.colorbar(im, fraction=0.046)
    _save(fig, "fig_confusion.pdf")


def fig_fusion_compare(values: dict):
    """values: {'text-only':.., 'image-only':.., 'late fusion':.., 'cross-attn':..}"""
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ks = list(values); vs = [values[k] for k in ks]
    ax.bar(ks, vs, color="#4c72b0")
    ax.set_ylim(0.95, 1.0); ax.set_ylabel("OOF macro-F1")
    for i, v in enumerate(vs):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    _save(fig, "fig_fusion_compare.pdf")
