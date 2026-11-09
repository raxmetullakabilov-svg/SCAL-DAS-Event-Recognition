from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from .config import BenchmarkConfig

REFERENCE_SPLIT_COLUMNS = [
    "event_id", "class_name", "recording_id", "leakage_group_id", "split", "split_reason",
    "window_start", "window_end", "u_centroid", "benchmark_version",
]


def _temporal_distance_windows(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Number of completely blank windows between two closed temporal intervals; 0 if overlapping/touching."""
    if a_end < b_start:
        return max(0, int(b_start) - int(a_end) - 1)
    if b_end < a_start:
        return max(0, int(a_start) - int(b_end) - 1)
    return 0


def _spatial_intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(int(a_start), int(b_start)) <= min(int(a_end), int(b_end))


def _stable_leakage_group_id(event_ids: list[str]) -> str:
    payload = "|".join(sorted(map(str, event_ids))).encode("utf-8")
    return "LG_" + hashlib.sha256(payload).hexdigest()[:16]


def build_spatiotemporal_leakage_groups(events: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    """Assign deterministic leakage groups using time proximity AND spatial overlap within each recording.

    Two events conflict when they originate from the same recording, have fewer than
    ``temporal_guard_windows`` blank windows between their closed temporal intervals,
    and their raw channel intervals overlap. Connected conflict components are atomic
    units for the B1 split.
    """
    required = {
        "event_id", "class_name", "recording_id", "window_start", "window_end",
        "channel_start", "channel_end",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"event registry missing required columns for leakage grouping: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("event registry contains duplicate event_id values")
    if events.empty:
        return pd.DataFrame(columns=["event_id", "leakage_group_id", "leakage_group_size"])

    work = events.reset_index(drop=True).copy()
    parent = list(range(len(work)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # deterministic representative: smaller row index wins
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    guard = int(config.temporal_guard_windows)
    for _, rec_group in work.groupby("recording_id", sort=True):
        idxs = rec_group.index.tolist()
        for pos, i in enumerate(idxs):
            a = work.loc[i]
            for j in idxs[pos + 1:]:
                b = work.loc[j]
                temporal_close = _temporal_distance_windows(
                    int(a["window_start"]), int(a["window_end"]),
                    int(b["window_start"]), int(b["window_end"]),
                ) < guard
                if not temporal_close:
                    continue
                if _spatial_intervals_overlap(
                    int(a["channel_start"]), int(a["channel_end"]),
                    int(b["channel_start"]), int(b["channel_end"]),
                ):
                    union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(len(work)):
        components.setdefault(find(i), []).append(i)

    rows: list[dict] = []
    for indices in components.values():
        classes = sorted(work.loc[indices, "class_name"].astype(str).unique().tolist())
        if len(classes) != 1:
            event_ids = work.loc[indices, "event_id"].astype(str).tolist()
            raise ValueError(
                "a spatio-temporal leakage group spans multiple class labels; "
                f"cannot perform class-stratified B1 split safely: classes={classes}, events={event_ids[:20]}"
            )
        event_ids = work.loc[indices, "event_id"].astype(str).tolist()
        group_id = _stable_leakage_group_id(event_ids)
        for event_id in event_ids:
            rows.append({
                "event_id": event_id,
                "leakage_group_id": group_id,
                "leakage_group_size": len(event_ids),
            })
    return pd.DataFrame(rows).sort_values("event_id", kind="stable").reset_index(drop=True)


def _ordered_class_leakage_groups(class_events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for group_id, group in class_events.groupby("leakage_group_id", sort=True):
        dates = group.get("recording_date", pd.Series([""] * len(group), index=group.index)).fillna("").astype(str)
        rows.append({
            "leakage_group_id": str(group_id),
            "recording_date": dates.min() if len(dates) else "",
            "recording_id": str(group["recording_id"].astype(str).min()),
            "window_start": int(group["window_start"].min()),
            "window_end": int(group["window_end"].max()),
            "event_count": int(group["event_id"].nunique()),
        })
    return pd.DataFrame(rows).sort_values(
        ["recording_date", "recording_id", "window_start", "window_end", "leakage_group_id"],
        kind="stable",
    ).reset_index(drop=True)


def _choose_group_cuts(group_table: pd.DataFrame, config: BenchmarkConfig, class_name: str) -> tuple[int, int]:
    n_groups = len(group_table)
    if n_groups < 3:
        raise ValueError(
            f"class {class_name!r} has only {n_groups} independent spatio-temporal leakage groups; "
            "at least 3 are required for train/val/test coverage"
        )
    weights = group_table["event_count"].to_numpy(dtype=float)
    total_events = float(weights.sum())
    target = np.asarray(config.reference_split, dtype=float) * total_events
    prefix = np.concatenate([[0.0], np.cumsum(weights)])
    candidates: list[tuple[float, int, int, float, float, float]] = []
    for i in range(1, n_groups - 1):
        for j in range(i + 1, n_groups):
            counts = np.asarray([prefix[i], prefix[j] - prefix[i], prefix[-1] - prefix[j]], dtype=float)
            score = float(np.abs(counts - target).sum())
            # deterministic tie breakers: closer train target, then balanced val/test, then earliest cuts
            candidates.append((score, abs(counts[0] - target[0]), abs(counts[1] - counts[2]), i, j, counts[2]))
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], -x[5]))
    _, _, _, i, j, _ = candidates[0]
    return int(i), int(j)


def build_reference_split(events: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    required = {
        "event_id", "class_name", "recording_id", "window_start", "window_end",
        "channel_start", "channel_end", "u_centroid",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"event registry missing required columns: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("event registry contains duplicate event_id values")

    mapping = build_spatiotemporal_leakage_groups(events, config)
    work = events.merge(mapping[["event_id", "leakage_group_id"]], on="event_id", how="left", validate="one_to_one")
    rows: list[dict] = []
    for class_name, class_events in work.groupby("class_name", sort=True):
        group_table = _ordered_class_leakage_groups(class_events)
        train_cut, val_cut = _choose_group_cuts(group_table, config, str(class_name))
        group_split: dict[str, str] = {}
        for idx, row in group_table.iterrows():
            if idx < train_cut:
                split = "train"
            elif idx < val_cut:
                split = "val"
            else:
                split = "test"
            group_split[str(row["leakage_group_id"])] = split

        ordered_events = class_events.sort_values(
            ["recording_date", "recording_id", "window_start", "window_end", "event_id"],
            kind="stable",
        ) if "recording_date" in class_events.columns else class_events.sort_values(
            ["recording_id", "window_start", "window_end", "event_id"], kind="stable"
        )
        for _, row in ordered_events.iterrows():
            gid = str(row["leakage_group_id"])
            rows.append({
                "event_id": row["event_id"],
                "class_name": row["class_name"],
                "recording_id": row["recording_id"],
                "leakage_group_id": gid,
                "split": group_split[gid],
                "split_reason": (
                    "class-stratified deterministic split of atomic spatio-temporal leakage groups; "
                    f"events < {config.temporal_guard_windows} blank windows apart are grouped only when spatial channel intervals overlap"
                ),
                "window_start": int(row["window_start"]),
                "window_end": int(row["window_end"]),
                "u_centroid": float(row["u_centroid"]),
                "benchmark_version": config.benchmark_version,
            })
    return pd.DataFrame(rows, columns=REFERENCE_SPLIT_COLUMNS)

TRANSFER_SPLIT_COLUMNS = [
    "event_id", "class_name", "recording_id", "leakage_group_id", "spatial_fold", "transfer_pair_id",
    "transfer_role", "u_start", "u_end", "u_centroid", "eligibility_reason",
    "benchmark_version",
]


def assign_spatial_bins(events: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if "u_centroid" not in events.columns:
        raise ValueError("events must contain u_centroid")
    out = events.copy()
    u = out["u_centroid"].astype(float)
    if ((u < 0) | (u > 1)).any():
        raise ValueError("u_centroid values must lie in [0,1]")
    folds = np.floor(u.to_numpy() * n_bins).astype(int) + 1
    folds = np.clip(folds, 1, n_bins)
    out["spatial_fold"] = folds
    return out


def _crosses_spatial_boundary(u_start: float, u_end: float, n_folds: int) -> bool:
    for k in range(1, n_folds):
        boundary = k / n_folds
        if float(u_start) < boundary < float(u_end):
            return True
    return False


def _eligible_classes_for_pair(
    events: pd.DataFrame,
    source_fold: int,
    target_fold: int,
    config: BenchmarkConfig,
) -> set[str]:
    eligible: set[str] = set()
    for cls, group in events.groupby("class_name", sort=True):
        source = group[group["spatial_fold"] == source_fold]
        target = group[group["spatial_fold"] == target_fold]
        source_count = int(source["event_id"].nunique())
        target_count = int(target["event_id"].nunique())
        source_groups = int(source["leakage_group_id"].nunique()) if "leakage_group_id" in source.columns else 0
        if (
            source_count >= config.min_transfer_events_per_class
            and target_count >= config.min_transfer_events_per_class
            and source_groups >= config.min_transfer_source_groups_per_class
        ):
            eligible.add(str(cls))
    return eligible


def _assign_source_group_roles(src: pd.DataFrame) -> dict[str, str]:
    """Deterministically assign whole leakage groups to source_train/source_val."""
    group_table = _ordered_class_leakage_groups(src)
    n_groups = len(group_table)
    if n_groups < 2:
        raise ValueError("source split requires at least two independent leakage groups")
    n_train_groups = max(1, int(math.floor(n_groups * 0.80)))
    if n_train_groups >= n_groups:
        n_train_groups = n_groups - 1
    train_groups = set(group_table.iloc[:n_train_groups]["leakage_group_id"].astype(str))
    role_by_event: dict[str, str] = {}
    for row in src.itertuples(index=False):
        role_by_event[str(row.event_id)] = (
            "source_train" if str(row.leakage_group_id) in train_groups else "source_val"
        )
    return role_by_event

def build_spatial_transfer_splits(events: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    required = {"event_id", "class_name", "recording_id", "window_start", "window_end", "channel_start", "channel_end", "u_start", "u_end", "u_centroid"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"event registry missing required columns for spatial transfer: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("event registry contains duplicate event_id values")

    work = assign_spatial_bins(events, config.n_spatial_folds)
    work["boundary_crossing"] = [
        _crosses_spatial_boundary(a, b, config.n_spatial_folds)
        for a, b in zip(work["u_start"], work["u_end"])
    ]
    # Reuse the exact B1 spatio-temporal leakage definition.  B3 source
    # train/validation is split only at leakage-group boundaries.
    leakage = build_spatiotemporal_leakage_groups(events, config)
    work = work.merge(
        leakage[["event_id", "leakage_group_id"]],
        on="event_id", how="left", validate="one_to_one",
    )
    support = work[~work["boundary_crossing"]].copy()
    rows: list[dict] = []

    pair_eligibility: dict[tuple[int, int], set[str]] = {}
    for source_fold in range(1, config.n_spatial_folds + 1):
        for target_fold in range(1, config.n_spatial_folds + 1):
            if source_fold == target_fold:
                continue
            eligible = _eligible_classes_for_pair(support, source_fold, target_fold, config)
            if len(eligible) >= config.min_transfer_classes:
                pair_eligibility[(source_fold, target_fold)] = eligible

    preferred_forward = (1, config.n_spatial_folds)
    preferred_reverse = (config.n_spatial_folds, 1)
    if preferred_forward in pair_eligibility and preferred_reverse in pair_eligibility:
        selected_pairs = [preferred_forward, preferred_reverse]
    else:
        selected_pairs = sorted(pair_eligibility)

    for source_fold, target_fold in selected_pairs:
            eligible = pair_eligibility[(source_fold, target_fold)]
            pair_id = f"F{source_fold}_to_F{target_fold}"
            relevant = work[
                work["boundary_crossing"]
                | work["spatial_fold"].isin([source_fold, target_fold])
            ].copy()

            source_role: dict[str, str] = {}
            for cls in sorted(eligible):
                src = relevant[
                    (~relevant["boundary_crossing"])
                    & (relevant["class_name"] == cls)
                    & (relevant["spatial_fold"] == source_fold)
                ].sort_values(["recording_id", "window_start", "event_id"], kind="stable")
                source_role.update(_assign_source_group_roles(src))

            for row in relevant.sort_values(["class_name", "recording_id", "window_start", "event_id"], kind="stable").itertuples(index=False):
                if bool(row.boundary_crossing):
                    role = "excluded_boundary"
                    reason = "component crosses normalized spatial-fold boundary"
                elif row.class_name not in eligible:
                    role = "ineligible_class"
                    reason = (
                        f"class lacks minimum {config.min_transfer_events_per_class} events in both regions "
                        f"or {config.min_transfer_source_groups_per_class} independent source leakage groups "
                        f"for source F{source_fold} and target F{target_fold}"
                    )
                elif int(row.spatial_fold) == target_fold:
                    role = "target_test"
                    reason = "eligible class in target fold"
                elif int(row.spatial_fold) == source_fold:
                    role = source_role[str(row.event_id)]
                    reason = "eligible class in source fold; deterministic 80/20 split of atomic leakage groups"
                else:
                    continue
                rows.append(
                    {
                        "event_id": row.event_id,
                        "class_name": row.class_name,
                        "recording_id": row.recording_id,
                        "leakage_group_id": str(row.leakage_group_id),
                        "spatial_fold": int(row.spatial_fold),
                        "transfer_pair_id": pair_id,
                        "transfer_role": role,
                        "u_start": float(row.u_start),
                        "u_end": float(row.u_end),
                        "u_centroid": float(row.u_centroid),
                        "eligibility_reason": reason,
                        "benchmark_version": config.benchmark_version,
                    }
                )
    return pd.DataFrame(rows, columns=TRANSFER_SPLIT_COLUMNS)
