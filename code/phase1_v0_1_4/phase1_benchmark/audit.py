from __future__ import annotations

from collections import Counter

import pandas as pd

from .config import BenchmarkConfig
from .splits import build_reference_split, build_spatiotemporal_leakage_groups

AUDIT_COLUMNS = ["gate", "status", "detail"]


def _gate(name: str, ok: bool, detail: str) -> dict:
    return {"gate": name, "status": "PASS" if ok else "FAIL", "detail": detail}


def audit_event_registry(events: pd.DataFrame, source_inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    dup = sorted(events.loc[events["event_id"].duplicated(keep=False), "event_id"].astype(str).unique().tolist()) if "event_id" in events else ["missing event_id"]
    rows.append(_gate("registry_unique_event_id", len(dup) == 0, "no duplicate event IDs" if not dup else f"duplicates: {dup[:20]}"))

    geometry_cols = {"window_start", "window_end", "channel_start", "channel_end", "component_area_cells"}
    if geometry_cols.issubset(events.columns):
        bad = events[(events["window_start"] > events["window_end"]) | (events["channel_start"] > events["channel_end"]) | (events["component_area_cells"] <= 0)]
        rows.append(_gate("registry_valid_geometry", bad.empty, "all component geometries valid" if bad.empty else f"invalid events: {bad['event_id'].astype(str).tolist()[:20]}"))
    else:
        rows.append(_gate("registry_valid_geometry", False, f"missing columns: {sorted(geometry_cols.difference(events.columns))}"))

    if {"u_start", "u_end", "u_centroid"}.issubset(events.columns):
        bad_u = events[(events["u_start"] < 0) | (events["u_end"] > 1) | (events["u_start"] > events["u_end"]) | (events["u_centroid"] < events["u_start"]) | (events["u_centroid"] > events["u_end"])]
        rows.append(_gate("registry_normalized_coordinates", bad_u.empty, "all normalized coordinates in [0,1]" if bad_u.empty else f"invalid events: {bad_u['event_id'].astype(str).tolist()[:20]}"))
    else:
        rows.append(_gate("registry_normalized_coordinates", False, "missing normalized-coordinate columns"))

    if {"source_mask_file", "source_mask_sha256"}.issubset(events.columns) and {"source_mask_file", "source_mask_sha256"}.issubset(source_inventory.columns):
        inv = source_inventory.drop_duplicates("source_mask_file").set_index("source_mask_file")["source_mask_sha256"].astype(str).to_dict()
        missing = []
        mismatch = []
        for row in events[["source_mask_file", "source_mask_sha256"]].drop_duplicates().itertuples(index=False):
            if row.source_mask_file not in inv:
                missing.append(str(row.source_mask_file))
            elif str(inv[row.source_mask_file]) != str(row.source_mask_sha256):
                mismatch.append(str(row.source_mask_file))
        rows.append(_gate("registry_source_inventory", not missing and not mismatch, "all source masks present with matching recorded checksums" if not missing and not mismatch else f"missing={missing[:10]}, checksum_mismatch={mismatch[:10]}"))
    else:
        rows.append(_gate("registry_source_inventory", False, "source inventory or event source columns missing"))
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def audit_reference_split(split_df: pd.DataFrame, events: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    rows: list[dict] = []
    kept = split_df[split_df["split"].isin(["train", "val", "test"])] if {"split", "event_id"}.issubset(split_df.columns) else pd.DataFrame(columns=["event_id", "split"])
    by_event = kept.groupby("event_id")["split"].agg(lambda s: sorted(set(map(str, s)))) if not kept.empty else pd.Series(dtype=object)
    overlap = sorted([str(eid) for eid, splits in by_event.items() if len(splits) > 1])
    rows.append(_gate("reference_pairwise_event_disjoint", len(overlap) == 0, "train/val/test event sets are disjoint" if not overlap else f"events assigned across multiple splits: {overlap[:20]}"))

    duplicate_assignments = sorted(split_df.loc[split_df["event_id"].duplicated(keep=False), "event_id"].astype(str).unique().tolist()) if "event_id" in split_df else []
    rows.append(_gate("reference_one_assignment_per_event", len(duplicate_assignments) == 0, "one split row per event" if not duplicate_assignments else f"duplicate split rows: {duplicate_assignments[:20]}"))

    group_required = {"event_id", "class_name", "recording_id", "window_start", "window_end", "channel_start", "channel_end"}
    if group_required.issubset(events.columns) and {"event_id", "split"}.issubset(split_df.columns):
        try:
            grouping = build_spatiotemporal_leakage_groups(events, config)
            assigned_groups = split_df[["event_id", "split"]].merge(grouping, on="event_id", how="left")
            crossing = []
            for group_id, group in assigned_groups.groupby("leakage_group_id", dropna=False, sort=True):
                active_splits = sorted(set(group.loc[group["split"].isin(["train", "val", "test"]), "split"].astype(str)))
                if len(active_splits) > 1:
                    crossing.append({
                        "leakage_group_id": str(group_id),
                        "splits": active_splits,
                        "event_ids": sorted(group["event_id"].astype(str).tolist()),
                    })
            detail = "every spatio-temporal leakage group is atomic within one B1 partition" if not crossing else f"groups crossing partitions: {crossing[:10]}"
            rows.append(_gate("reference_leakage_group_disjoint", len(crossing) == 0, detail))
        except ValueError as exc:
            rows.append(_gate("reference_leakage_group_disjoint", False, f"cannot construct leakage groups: {exc}"))
    else:
        rows.append(_gate("reference_leakage_group_disjoint", False, f"cannot verify leakage groups; missing columns: {sorted(group_required.difference(events.columns))}"))

    if {"event_id", "split"}.issubset(split_df.columns) and {"event_id", "class_name"}.issubset(events.columns):
        assigned = split_df[["event_id", "split"]].merge(events[["event_id", "class_name"]], on="event_id", how="left")
        all_classes = set(events["class_name"].astype(str))
        train_classes = set(assigned.loc[assigned["split"] == "train", "class_name"].dropna().astype(str))
        val_classes = set(assigned.loc[assigned["split"] == "val", "class_name"].dropna().astype(str))
        test_classes = set(assigned.loc[assigned["split"] == "test", "class_name"].dropna().astype(str))
        missing_train = sorted(all_classes - train_classes)
        missing_val = sorted(all_classes - val_classes)
        missing_test = sorted(all_classes - test_classes)
        ok_coverage = not missing_train and not missing_val and not missing_test
        detail = "every registry class appears in train, validation, and test" if ok_coverage else f"missing_train={missing_train}, missing_val={missing_val}, missing_test={missing_test}"
        rows.append(_gate("reference_train_val_test_class_coverage", ok_coverage, detail))
    else:
        rows.append(_gate("reference_train_val_test_class_coverage", False, "cannot verify class coverage; event_id/split/class_name columns missing"))

    if set(events["event_id"].astype(str)) == set(split_df["event_id"].astype(str)) and {"class_name", "recording_id", "window_start", "window_end", "u_centroid"}.issubset(events.columns):
        try:
            expected = build_reference_split(events, config).set_index("event_id")["split"].astype(str).to_dict()
            actual = split_df.drop_duplicates("event_id").set_index("event_id")["split"].astype(str).to_dict()
            mismatch = sorted([eid for eid in expected if actual.get(eid) != expected[eid]])
            rows.append(_gate("reference_spatiotemporal_group_rebuild", len(mismatch) == 0, "assignments match deterministic class-stratified spatio-temporal leakage-group rebuild" if not mismatch else f"mismatched assignments: {mismatch[:20]}"))
        except ValueError as exc:
            rows.append(_gate("reference_spatiotemporal_group_rebuild", False, f"deterministic spatio-temporal rebuild infeasible: {exc}"))
    else:
        missing = sorted(set(events["event_id"].astype(str)) - set(split_df["event_id"].astype(str)))
        rows.append(_gate("reference_spatiotemporal_group_rebuild", False, f"cannot verify deterministic rebuild; missing events or columns; missing_event_ids={missing[:20]}"))
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def _crosses_boundary(a: float, b: float, n_folds: int) -> bool:
    return any(float(a) < k / n_folds < float(b) for k in range(1, n_folds))


def audit_spatial_transfer(transfer_df: pd.DataFrame, events: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    rows: list[dict] = []
    valid_pairs = []
    if not transfer_df.empty and {"transfer_pair_id", "class_name", "transfer_role"}.issubset(transfer_df.columns):
        for pair_id, group in transfer_df.groupby("transfer_pair_id", sort=True):
            active = group[group["transfer_role"].isin(["source_train", "source_val", "target_test"])]
            classes = sorted(active["class_name"].astype(str).unique().tolist())
            if len(classes) >= config.min_transfer_classes:
                valid_pairs.append(str(pair_id))
    rows.append(_gate(
        "transfer_valid_pair_available",
        len(valid_pairs) > 0,
        f"valid transfer pairs: {valid_pairs}" if valid_pairs else (
            f"no spatial pair has at least {config.min_transfer_classes} classes with "
            f">= {config.min_transfer_events_per_class} events in both source and target"
        ),
    ))
    overlaps: set[str] = set()
    if not transfer_df.empty and {"transfer_pair_id", "transfer_role", "event_id"}.issubset(transfer_df.columns):
        for _, group in transfer_df.groupby("transfer_pair_id", sort=True):
            src = set(group.loc[group["transfer_role"].isin(["source_train", "source_val"]), "event_id"].astype(str))
            tgt = set(group.loc[group["transfer_role"] == "target_test", "event_id"].astype(str))
            overlaps.update(src & tgt)
    rows.append(_gate("transfer_source_target_event_disjoint", len(overlaps) == 0, "source/target event sets are disjoint within every transfer pair" if not overlaps else f"overlap events: {sorted(overlaps)[:20]}"))

    # Source train/validation must respect the same atomic spatio-temporal leakage groups as B1.
    source_group_crossings: list[str] = []
    needed = {"event_id", "class_name", "recording_id", "window_start", "window_end", "channel_start", "channel_end"}
    if not transfer_df.empty and needed.issubset(events.columns):
        leakage = build_spatiotemporal_leakage_groups(events, config)
        active_source = transfer_df[transfer_df["transfer_role"].isin(["source_train", "source_val"])][
            ["event_id", "transfer_pair_id", "transfer_role"]
        ].merge(leakage[["event_id", "leakage_group_id"]], on="event_id", how="left", validate="many_to_one")
        for (pair_id, group_id), grp in active_source.groupby(["transfer_pair_id", "leakage_group_id"], sort=True):
            roles = sorted(grp["transfer_role"].astype(str).unique().tolist())
            if len(roles) > 1:
                source_group_crossings.append(f"{pair_id}:{group_id}:{roles}")
        rows.append(_gate(
            "transfer_source_train_val_leakage_group_disjoint",
            len(source_group_crossings) == 0,
            "source train/validation preserve atomic leakage groups" if not source_group_crossings else f"source leakage groups cross train/val: {source_group_crossings[:20]}",
        ))
    else:
        rows.append(_gate(
            "transfer_source_train_val_leakage_group_disjoint",
            False if not transfer_df.empty else True,
            "no active transfer rows" if transfer_df.empty else "cannot verify source train/validation leakage groups; event geometry columns missing",
        ))

    bad_boundary: list[str] = []
    if not transfer_df.empty and {"u_start", "u_end", "transfer_role", "event_id"}.issubset(transfer_df.columns):
        for row in transfer_df.itertuples(index=False):
            if _crosses_boundary(row.u_start, row.u_end, config.n_spatial_folds) and row.transfer_role in {"source_train", "source_val", "target_test"}:
                bad_boundary.append(str(row.event_id))
    rows.append(_gate("transfer_boundary_events_excluded", len(bad_boundary) == 0, "no boundary-crossing event assigned to source/target" if not bad_boundary else f"boundary-assigned events: {sorted(set(bad_boundary))[:20]}"))

    support_failures: list[str] = []
    if not transfer_df.empty:
        for (pair_id, cls), group in transfer_df.groupby(["transfer_pair_id", "class_name"], sort=True):
            src_n = group.loc[group["transfer_role"].isin(["source_train", "source_val"]), "event_id"].nunique()
            tgt_n = group.loc[group["transfer_role"] == "target_test", "event_id"].nunique()
            if (src_n > 0 or tgt_n > 0) and (src_n < config.min_transfer_events_per_class or tgt_n < config.min_transfer_events_per_class):
                support_failures.append(f"{pair_id}:{cls}:{src_n}/{tgt_n}")
    rows.append(_gate("transfer_minimum_class_support", len(support_failures) == 0, "all assigned eligible classes satisfy minimum support" if not support_failures else f"insufficient support: {support_failures[:20]}"))
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)
