from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .components import sha256_file
from .config import BenchmarkConfig


def _jsonable(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_dataframe(df: pd.DataFrame) -> list[dict]:
    if df is None:
        return []
    columns = sorted(map(str, df.columns))
    records = []
    for _, row in df.iterrows():
        record = {col: _jsonable(row[col]) for col in columns}
        records.append(record)
    records.sort(key=lambda r: json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return records


def _payload(
    config: BenchmarkConfig,
    source_inventory: pd.DataFrame,
    events: pd.DataFrame,
    reference_split: pd.DataFrame,
    transfer_split: pd.DataFrame,
) -> dict:
    return {
        "config": config.to_dict(),
        "code_version": config.code_version,
        "sources": _canonical_dataframe(source_inventory[["source_mask_file", "source_mask_sha256"]] if not source_inventory.empty else source_inventory),
        "event_registry": _canonical_dataframe(events),
        "reference_split": _canonical_dataframe(reference_split),
        "spatial_transfer_split": _canonical_dataframe(transfer_split),
    }


def benchmark_fingerprint(
    config: BenchmarkConfig,
    source_inventory: pd.DataFrame,
    events: pd.DataFrame,
    reference_split: pd.DataFrame,
    transfer_split: pd.DataFrame,
) -> str:
    raw = json.dumps(_payload(config, source_inventory, events, reference_split, transfer_split), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()



def build_persisted_file_hashes(root: Path, relative_paths: list[str]) -> dict[str, str]:
    """Hash exact persisted bytes using paths relative to benchmark root."""
    root = Path(root)
    hashes: dict[str, str] = {}
    for rel in sorted({Path(x).as_posix() for x in relative_paths}):
        path = root / rel
        if not path.is_file():
            raise ValueError(f"frozen artifact missing: {rel}")
        hashes[rel] = sha256_file(path)
    return hashes


def verify_persisted_file_hashes(root: Path, file_hashes: dict[str, str]) -> None:
    root = Path(root)
    for rel, expected in sorted(file_hashes.items()):
        path = root / rel
        if not path.is_file():
            raise ValueError(f"frozen artifact missing: {rel}")
        actual = sha256_file(path)
        if actual != str(expected):
            raise ValueError(f"persisted artifact checksum mismatch for {rel}: expected {expected}, got {actual}")


def persisted_benchmark_fingerprint(config: BenchmarkConfig, file_hashes: dict[str, str]) -> str:
    payload = {
        "benchmark_version": config.benchmark_version,
        "code_version": config.code_version,
        "config": config.to_dict(),
        "file_hashes": {k: file_hashes[k] for k in sorted(file_hashes)},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def build_manifest(
    config: BenchmarkConfig,
    source_inventory: pd.DataFrame,
    events: pd.DataFrame,
    reference_split: pd.DataFrame,
    transfer_split: pd.DataFrame,
    *,
    file_hashes: dict[str, str] | None = None,
) -> dict:
    sources = _canonical_dataframe(source_inventory[["source_mask_file", "source_mask_sha256"]] if not source_inventory.empty else source_inventory)
    if file_hashes is None:
        benchmark_sha = benchmark_fingerprint(config, source_inventory, events, reference_split, transfer_split)
        fingerprint_basis = "in_memory_dataframe_payload"
    else:
        benchmark_sha = persisted_benchmark_fingerprint(config, file_hashes)
        fingerprint_basis = "persisted_artifact_bytes"
    return {
        "benchmark_version": config.benchmark_version,
        "code_version": config.code_version,
        "config": config.to_dict(),
        "sources": sources,
        "counts": {
            "events": int(len(events)),
            "reference_rows": int(len(reference_split)),
            "transfer_rows": int(len(transfer_split)),
        },
        "fingerprint_basis": fingerprint_basis,
        "file_hashes": {k: file_hashes[k] for k in sorted(file_hashes)} if file_hashes is not None else None,
        "benchmark_sha256": benchmark_sha,
    }


def verify_frozen_benchmark(
    manifest: dict,
    source_root: Path | None = None,
    *,
    config: BenchmarkConfig | None = None,
    source_inventory: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    reference_split: pd.DataFrame | None = None,
    transfer_split: pd.DataFrame | None = None,
    benchmark_root: Path | None = None,
) -> None:
    root = Path(source_root) if source_root is not None else None
    for source in manifest.get("sources", []):
        p = Path(source["source_mask_file"])
        if root is not None and not p.is_absolute():
            p = root / p
        if not p.exists():
            raise ValueError(f"source mask missing: {p}")
        actual = sha256_file(p)
        expected = str(source["source_mask_sha256"])
        if actual != expected:
            raise ValueError(f"checksum mismatch for source mask {p}: expected {expected}, got {actual}")

    if manifest.get("file_hashes") is not None:
        if benchmark_root is None:
            raise ValueError("benchmark_root is required to verify persisted artifact hashes")
        verify_persisted_file_hashes(Path(benchmark_root), manifest["file_hashes"])
        if config is not None:
            persisted_fp = persisted_benchmark_fingerprint(config, manifest["file_hashes"])
            if persisted_fp != manifest.get("benchmark_sha256"):
                raise ValueError(f"persisted benchmark fingerprint mismatch: expected {manifest.get('benchmark_sha256')}, got {persisted_fp}")

    supplied = [config, source_inventory, events, reference_split, transfer_split]
    if any(x is not None for x in supplied):
        if any(x is None for x in supplied):
            raise ValueError("config, source_inventory, events, reference_split, and transfer_split must all be supplied together")
        if manifest.get("fingerprint_basis") != "persisted_artifact_bytes":
            actual_fp = benchmark_fingerprint(config, source_inventory, events, reference_split, transfer_split)  # type: ignore[arg-type]
            if actual_fp != manifest.get("benchmark_sha256"):
                raise ValueError(f"benchmark fingerprint mismatch: expected {manifest.get('benchmark_sha256')}, got {actual_fp}")
