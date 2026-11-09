from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_event_component_examples(events: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not events.empty:
        for cls, group in events.groupby("class_name", sort=True):
            ax.scatter(group["window_centroid"], group["u_centroid"], label=str(cls), alpha=0.8)
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Window centroid")
    ax.set_ylabel("Normalized fiber coordinate, u")
    ax.set_title("Connected event components")
    return _save(fig, path)


def plot_class_split_distribution(reference_split: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not reference_split.empty:
        pivot = reference_split.pivot_table(index="class_name", columns="split", values="event_id", aggfunc="nunique", fill_value=0)
        pivot.plot(kind="bar", ax=ax)
        ax.legend(title="Split", fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of event components")
    ax.set_title("Reference benchmark split distribution")
    return _save(fig, path)


def plot_spatial_transfer_map(transfer_split: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not transfer_split.empty:
        sample = transfer_split.reset_index(drop=True)
        for role, group in sample.groupby("transfer_role", sort=True):
            ax.scatter(group.index, group["u_centroid"], label=str(role), alpha=0.8)
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Transfer-row index")
    ax.set_ylabel("Normalized fiber coordinate, u")
    ax.set_title("Spatial transfer benchmark map")
    return _save(fig, path)
