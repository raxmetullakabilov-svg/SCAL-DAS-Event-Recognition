from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import BenchmarkConfig

EVENT_REGISTRY_COLUMNS = [
    "benchmark_version", "event_id", "class_id", "class_name", "recording_id",
    "recording_date", "source_mask_file", "source_mask_sha256", "component_index",
    "component_area_cells", "source_n_windows", "window_start", "window_end", "n_windows",
    "channel_start", "channel_end", "n_channels_span", "window_centroid",
    "channel_centroid", "spatial_dim", "u_start", "u_end", "u_centroid",
    "touches_left_edge", "touches_right_edge", "touches_first_window",
    "touches_last_window", "candidate_status", "exclusion_reason",
]


def validate_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a NumPy array")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D [windows, spatial]; got shape {mask.shape}")
    if mask.shape[1] < 2:
        raise ValueError("mask spatial dimension must contain at least two channels")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_event_id(
    *,
    class_name: str,
    recording_id: str,
    component_index: int,
    window_start: int,
    window_end: int,
    channel_start: int,
    channel_end: int,
) -> str:
    payload = "|".join(
        map(
            str,
            [
                class_name,
                recording_id,
                component_index,
                window_start,
                window_end,
                channel_start,
                channel_end,
            ],
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _connectivity_structure(connectedness: int) -> np.ndarray:
    if connectedness != 8:
        raise ValueError("Phase1_v0.1 requires 8-connectivity")
    return np.ones((3, 3), dtype=bool)


def extract_components(
    *,
    mask: np.ndarray,
    class_name: str,
    class_id: int,
    recording_id: str,
    recording_date: str | None,
    source_mask_file: str,
    source_mask_sha256: str,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    mask = np.asarray(mask)
    validate_mask(mask)
    mask_bool = mask.astype(bool, copy=False)
    n_time, n_spatial = mask_bool.shape
    labeled, n_components = ndimage.label(mask_bool, structure=_connectivity_structure(config.connectedness))

    rows: list[dict] = []
    object_slices = ndimage.find_objects(labeled, max_label=n_components)
    for component_index in range(1, n_components + 1):
        component_slice = object_slices[component_index - 1]
        if component_slice is None:
            continue
        local_labels = labeled[component_slice]
        local_coords = np.argwhere(local_labels == component_index)
        if local_coords.size == 0:
            continue
        windows = local_coords[:, 0] + int(component_slice[0].start)
        channels = local_coords[:, 1] + int(component_slice[1].start)
        coords = np.column_stack([windows, channels])
        w0, w1 = int(windows.min()), int(windows.max())
        c0, c1 = int(channels.min()), int(channels.max())
        wc = float(windows.mean())
        cc = float(channels.mean())
        denom = float(n_spatial - 1)
        event_id = make_event_id(
            class_name=class_name,
            recording_id=recording_id,
            component_index=component_index,
            window_start=w0,
            window_end=w1,
            channel_start=c0,
            channel_end=c1,
        )
        rows.append(
            {
                "benchmark_version": config.benchmark_version,
                "event_id": event_id,
                "class_id": int(class_id),
                "class_name": class_name,
                "recording_id": recording_id,
                "recording_date": recording_date,
                "source_mask_file": source_mask_file,
                "source_mask_sha256": source_mask_sha256,
                "component_index": int(component_index),
                "component_area_cells": int(len(coords)),
                "source_n_windows": int(n_time),
                "window_start": w0,
                "window_end": w1,
                "n_windows": int(w1 - w0 + 1),
                "channel_start": c0,
                "channel_end": c1,
                "n_channels_span": int(c1 - c0 + 1),
                "window_centroid": wc,
                "channel_centroid": cc,
                "spatial_dim": int(n_spatial),
                "u_start": c0 / denom,
                "u_end": c1 / denom,
                "u_centroid": cc / denom,
                "touches_left_edge": bool(c0 == 0),
                "touches_right_edge": bool(c1 == n_spatial - 1),
                "touches_first_window": bool(w0 == 0),
                "touches_last_window": bool(w1 == n_time - 1),
                "candidate_status": "included",
                "exclusion_reason": "",
            }
        )

    return pd.DataFrame(rows, columns=EVENT_REGISTRY_COLUMNS)
