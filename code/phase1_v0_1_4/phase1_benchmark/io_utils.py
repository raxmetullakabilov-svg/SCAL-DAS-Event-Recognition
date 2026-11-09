from pathlib import Path
from typing import Dict

from .config import BenchmarkConfig


def create_scaffold(root: Path, config: BenchmarkConfig) -> Dict[str, Path]:
    """Create the immutable benchmark directory contract without deleting data."""
    root = Path(root)
    paths = {
        "root": root,
        "inputs_masks": root / "inputs" / "masks",
        "inputs_phase0b": root / "inputs" / "phase0b",
        "registry": root / "registry",
        "splits": root / "splits",
        "audits": root / "audits",
        "figures": root / "figures",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

import json
import os
import tempfile

import pandas as pd


def _atomic_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_name(target.name + ".tmp")


def atomic_write_csv(df: pd.DataFrame, target: Path) -> Path:
    target = Path(target)
    tmp = _atomic_path(target)
    df.to_csv(tmp, index=False)
    os.replace(tmp, target)
    return target


def atomic_write_json(data: dict, target: Path) -> Path:
    target = Path(target)
    tmp = _atomic_path(target)
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def atomic_write_text(text: str, target: Path) -> Path:
    target = Path(target)
    tmp = _atomic_path(target)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
    return target


def write_event_registry(events: pd.DataFrame, paths: Dict[str, Path]) -> dict[str, Path]:
    registry_path = atomic_write_csv(events, paths["registry"] / "EVENT_REGISTRY.csv")
    if events.empty:
        stats = pd.DataFrame(columns=["class_name", "recording_id", "n_components", "total_area_cells", "median_area_cells"])
    else:
        stats = (
            events.groupby(["class_name", "recording_id"], dropna=False)
            .agg(
                n_components=("event_id", "nunique"),
                total_area_cells=("component_area_cells", "sum"),
                median_area_cells=("component_area_cells", "median"),
            )
            .reset_index()
        )
    stats_path = atomic_write_csv(stats, paths["registry"] / "COMPONENT_STATS.csv")
    if "candidate_status" in events.columns:
        excluded = events[events["candidate_status"] != "included"].copy()
    else:
        excluded = events.iloc[0:0].copy()
    excluded_path = atomic_write_csv(excluded, paths["registry"] / "EXCLUDED_COMPONENTS.csv")
    return {"event_registry": registry_path, "component_stats": stats_path, "excluded_components": excluded_path}


def _combined_exclusions(events: pd.DataFrame, reference_split: pd.DataFrame, transfer_split: pd.DataFrame) -> pd.DataFrame:
    reasons: list[dict] = []
    if "candidate_status" in events.columns:
        for row in events[events["candidate_status"] != "included"].itertuples(index=False):
            reasons.append({"event_id": row.event_id, "scope": "registry", "reason": str(getattr(row, "exclusion_reason", "registry exclusion"))})
    if not reference_split.empty:
        guard = reference_split[reference_split["split"] == "guard_excluded"]
        for row in guard.itertuples(index=False):
            reasons.append({"event_id": row.event_id, "scope": "B1_guard", "reason": str(row.split_reason)})
    if not transfer_split.empty:
        boundary = transfer_split[transfer_split["transfer_role"] == "excluded_boundary"]
        for row in boundary.itertuples(index=False):
            reasons.append({"event_id": row.event_id, "scope": "B3_boundary", "reason": str(row.eligibility_reason)})
    if not reasons:
        out = events.iloc[0:0].copy()
        out["benchmark_exclusion_scope"] = pd.Series(dtype=str)
        out["benchmark_exclusion_reason"] = pd.Series(dtype=str)
        return out
    reason_df = pd.DataFrame(reasons)
    grouped = (
        reason_df.groupby("event_id", sort=True)
        .agg(
            benchmark_exclusion_scope=("scope", lambda x: ";".join(sorted(set(map(str, x))))),
            benchmark_exclusion_reason=("reason", lambda x: "; ".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    out = events.merge(grouped, on="event_id", how="inner")
    if "exclusion_reason" not in out.columns:
        out["exclusion_reason"] = out["benchmark_exclusion_reason"]
    else:
        out["exclusion_reason"] = out["exclusion_reason"].astype("object")
        blank = out["exclusion_reason"].isna() | (out["exclusion_reason"].astype(str).str.strip() == "")
        out.loc[blank, "exclusion_reason"] = out.loc[blank, "benchmark_exclusion_reason"].astype("object")
    return out


def write_split_artifacts(reference_split: pd.DataFrame, transfer_split: pd.DataFrame, paths: Dict[str, Path], events: pd.DataFrame | None = None) -> dict[str, Path]:
    reference_path = atomic_write_csv(reference_split, paths["splits"] / "TRAIN_VAL_TEST.csv")
    transfer_path = atomic_write_csv(transfer_split, paths["splits"] / "SPATIAL_TRANSFER_SPLITS.csv")
    if reference_split.empty:
        class_stats = pd.DataFrame(columns=["class_name", "split", "n_events"])
    else:
        class_stats = (
            reference_split.groupby(["class_name", "split"], dropna=False)["event_id"]
            .nunique()
            .rename("n_events")
            .reset_index()
            .sort_values(["class_name", "split"], kind="stable")
        )
    stats_path = atomic_write_csv(class_stats, paths["splits"] / "CLASS_SPLIT_STATS.csv")
    outputs = {"reference_split": reference_path, "spatial_transfer": transfer_path, "class_split_stats": stats_path}
    if events is not None:
        exclusions = _combined_exclusions(events, reference_split, transfer_split)
        outputs["excluded_components"] = atomic_write_csv(exclusions, paths["registry"] / "EXCLUDED_COMPONENTS.csv")
    return outputs


def write_audit_artifacts(audit_df: pd.DataFrame, paths: Dict[str, Path]) -> Path:
    return atomic_write_csv(audit_df, paths["audits"] / "LEAKAGE_AUDIT.csv")


def write_manifest_and_fingerprint(manifest: dict, paths: Dict[str, Path]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if manifest.get("file_hashes") is not None:
        outputs["file_hashes"] = atomic_write_json(
            {"file_hashes": manifest["file_hashes"]},
            paths["root"] / "BENCHMARK_FILE_HASHES.json",
        )
    outputs["manifest"] = atomic_write_json(manifest, paths["root"] / "BENCHMARK_MANIFEST.json")
    fp = str(manifest.get("benchmark_sha256", ""))
    outputs["fingerprint"] = atomic_write_text(fp + "\n", paths["root"] / "BENCHMARK_SHA256.txt")
    return outputs
