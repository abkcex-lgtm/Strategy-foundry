from __future__ import annotations

import gzip
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from app.engine import normalize_contract, _goal_activity_evidence

TARGET_UNIQUE_CANDIDATES = 500_000


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _strategy_key(item: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical({
        "strategy": item.get("strategy", {}),
        "contract": item.get("contract", {}),
    }).encode("utf-8")).hexdigest()


def _near_key(item: dict[str, Any]) -> tuple:
    metrics = item.get("development_metrics") or item.get("train_metrics") or item.get("metrics") or {}
    cfg = normalize_contract(item.get("contract") or {})
    try:
        distance = float(item.get("development_distance", item.get("distance", float("inf"))))
    except (TypeError, ValueError):
        distance = float("inf")
    return (
        0 if item.get("development_audit_status") == "MATCH" else 1,
        not _goal_activity_evidence(metrics, cfg),
        distance,
        -int(metrics.get("trades", 0) or 0),
    )


def aggregate(artifact_root: Path, output_dir: Path, target: int = TARGET_UNIQUE_CANDIDATES) -> dict[str, Any]:
    report_paths = sorted(artifact_root.glob("*/research-results/report.json"))
    if not report_paths:
        raise RuntimeError(f"No shard reports found under {artifact_root}")

    shards = []
    global_hashes: set[str] = set()
    total_worker_evaluations = 0
    family_counts: dict[str, int] = {}
    all_items: list[dict[str, Any]] = []
    zip_sources: list[Path] = []
    hash_artifact_hashes: list[str] = []

    for report_path in report_paths:
        base = report_path.parent.parent
        report = json.loads(report_path.read_text(encoding="utf-8"))
        hash_path = report_path.parent / "evaluated_genome_hashes.txt.gz"
        zip_path = report_path.parent / "validated_strategies.zip"
        if not hash_path.is_file():
            raise RuntimeError(f"Missing exact genome-hash artifact for shard: {base.name}")
        if not zip_path.is_file():
            raise RuntimeError(f"Missing strategy export artifact for shard: {base.name}")
        with gzip.open(hash_path, "rt", encoding="ascii") as stream:
            shard_hashes = {line.strip() for line in stream if line.strip()}
        if len(shard_hashes) != int(report.get("evaluated_unique_candidates", -1)):
            raise RuntimeError(
                f"Genome hash count does not match report for {base.name}: "
                f"{len(shard_hashes)} vs {report.get('evaluated_unique_candidates')}"
            )
        global_hashes.update(shard_hashes)
        hash_artifact_hashes.append(hashlib.sha256(hash_path.read_bytes()).hexdigest())
        total_worker_evaluations += int(report.get("evaluated_unique_candidates", 0))
        for family, count in (report.get("family_evaluated_counts") or {}).items():
            family_counts[family] = family_counts.get(family, 0) + int(count)
        all_items.extend(report.get("results") or [])
        zip_sources.append(zip_path)
        shards.append({
            "artifact": base.name,
            "source_commit": report.get("source_commit"),
            "requested_candidates": report.get("requested_candidates"),
            "evaluated_unique_candidates": report.get("evaluated_unique_candidates"),
            "candidate_search_complete": report.get("candidate_search_complete"),
            "outcome": report.get("outcome"),
            "dataset_sha256": (report.get("data") or {}).get("dataset_sha256"),
            "train_sha256": ((report.get("splits") or {}).get("train") or {}).get("sha256"),
            "development_sha256": ((report.get("splits") or {}).get("development") or {}).get("sha256"),
            "oos_sha256": ((report.get("splits") or {}).get("oos") or {}).get("sha256"),
        })

    # A worker must never silently use a different data split or source commit.
    for field in ("dataset_sha256", "train_sha256", "development_sha256", "oos_sha256"):
        values = {s.get(field) for s in shards}
        if len(values) != 1:
            raise RuntimeError(f"Research shards disagree on {field}: {sorted(str(v) for v in values)}")
    source_commits = {s.get("source_commit") for s in shards}
    if len(source_commits) != 1:
        raise RuntimeError(f"Research shards used different source commits: {sorted(str(v) for v in source_commits)}")

    # Deduplicate candidates returned by multiple workers by exact frozen strategy/config.
    by_key: dict[str, dict[str, Any]] = {}
    for item in all_items:
        key = _strategy_key(item)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = item
        elif _near_key(item) < _near_key(existing):
            by_key[key] = item
    unique_items = list(by_key.values())

    validated = [
        item for item in unique_items
        if item.get("development_passes_all_targets") is True
        and item.get("development_audit_status") == "MATCH"
        and item.get("oos_passes_all_targets") is True
        and item.get("oos_audit_status") == "MATCH"
        and item.get("full_data_audit_status") == "MATCH"
    ]
    delivery = validated if validated else sorted(unique_items, key=_near_key)[:5]
    if not delivery:
        raise RuntimeError("No auditable candidates were returned by any shard")
    if not validated and len(delivery) != 5:
        raise RuntimeError(f"Only {len(delivery)} near-misses available; refusing to claim an exact Top-5")

    workers_complete = all(s.get("candidate_search_complete") is True for s in shards)
    globally_complete = len(global_hashes) >= int(target) and workers_complete
    if validated:
        outcome = "VALIDATED_STRATEGIES_FOUND" if globally_complete else "VALIDATED_STRATEGIES_FOUND_SEARCH_INCOMPLETE"
    else:
        outcome = "NO_STRATEGY_PASSED_ALL_REQUIRED_SPLITS" if globally_complete else "INCOMPLETE_SEARCH_NO_CERTIFIED_STRATEGY"

    # Find exact standalone export and ledger files for the chosen strategies.
    export_index: dict[str, tuple[Path, str]] = {}
    for zip_path in zip_sources:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for name in archive.namelist():
                if not name.endswith("_manifest.json"):
                    continue
                manifest = json.loads(archive.read(name).decode("utf-8"))
                key = hashlib.sha256(_canonical({
                    "strategy": manifest.get("strategy", {}),
                    "contract": manifest.get("contract", {}),
                }).encode("utf-8")).hexdigest()
                export_index.setdefault(key, (zip_path, name[:-len("_manifest.json")]))

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_zip = output_dir / "validated_strategies.zip"
    exports = []
    with zipfile.ZipFile(combined_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        out.writestr(
            "README.txt",
            "Combined Strategy Foundry research artifact. Strategy selection is ranked on Development only; OOS is reported after selection and is never used to rank near-misses. No live orders are sent.",
        )
        with zipfile.ZipFile(zip_sources[0], "r") as source:
            if "market_data_manifest.json" in source.namelist():
                out.writestr("market_data_manifest.json", source.read("market_data_manifest.json"))
        suffixes = (
            ".py", "_train_ledger.json", "_development_ledger.json",
            "_full_ledger.json", "_oos_ledger.json", "_manifest.json",
        )
        for index, item in enumerate(delivery, 1):
            key = _strategy_key(item)
            source_entry = export_index.get(key)
            if source_entry is None:
                raise RuntimeError(f"Chosen strategy has no matching standalone export/manifest: {key}")
            source_zip, prefix = source_entry
            with zipfile.ZipFile(source_zip, "r") as source:
                target_prefix = f"strategy_{index:03d}"
                for suffix in suffixes:
                    original_name = prefix + suffix
                    if original_name not in source.namelist():
                        raise RuntimeError(f"Export artifact missing required file: {original_name}")
                    out.writestr(target_prefix + suffix, source.read(original_name))
            exports.append({
                "file": f"strategy_{index:03d}.py",
                "strategy": item.get("strategy"),
                "full_run_metrics": item.get("full_run_metrics") or item.get("metrics"),
                "full_run_hash_sha256": item.get("full_run_hash_sha256"),
                "full_data_audit_status": item.get("full_data_audit_status"),
            })

    first_report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "outcome": outcome,
        "source_commit": next(iter(source_commits)),
        "requested_candidates": int(target),
        "worker_requested_candidates_total": sum(int(s.get("requested_candidates") or 0) for s in shards),
        "evaluated_unique_candidates": len(global_hashes),
        "total_worker_evaluations_before_cross_worker_deduplication": total_worker_evaluations,
        "duplicate_genomes_across_workers": total_worker_evaluations - len(global_hashes),
        "candidate_search_complete": globally_complete,
        "workers_complete": workers_complete,
        "validated_strategy_count": len(validated),
        "returned_candidate_count": len(delivery),
        "near_miss_shortfall": max(0, 5 - len(delivery)) if not validated else 0,
        "family_evaluated_counts": family_counts,
        "shards": shards,
        "data": first_report.get("data"),
        "splits": first_report.get("splits"),
        "contract": first_report.get("contract"),
        "target_criteria": first_report.get("target_criteria"),
        "selection_policy": "Candidate search is deduplicated globally by exact genome hash. All passing strategies are returned; otherwise exactly five near-misses are ranked by Development-only audit/evidence/distance. OOS is not used for ranking.",
        "exports": exports,
        "results": delivery,
        "research_artifacts": {
            "evaluated_genome_hash_files_sha256": hash_artifact_hashes,
            "combined_unique_genome_hash_count": len(global_hashes),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with gzip.open(output_dir / "evaluated_genome_hashes.txt.gz", "wt", encoding="ascii") as stream:
        for genome_hash in sorted(global_hashes):
            stream.write(genome_hash + "\n")
    (output_dir / "progress.json").write_text(
        json.dumps({
            "status": "completed" if globally_complete else "partial_search_reported",
            "outcome": outcome,
            "requested_candidates": int(target),
            "evaluated_unique_candidates": len(global_hashes),
            "candidate_search_complete": globally_complete,
            "validated_strategy_count": len(validated),
            "returned_candidate_count": len(delivery),
            "report": "research-results/report.json",
            "strategy_zip": "research-results/validated_strategies.zip",
        }, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    summary = aggregate(root / "research-artifacts", root / "research-results")
    print(json.dumps({
        "outcome": summary["outcome"],
        "evaluated_unique_candidates": summary["evaluated_unique_candidates"],
        "validated_strategy_count": summary["validated_strategy_count"],
        "returned_candidate_count": summary["returned_candidate_count"],
        "candidate_search_complete": summary["candidate_search_complete"],
    }, ensure_ascii=False, sort_keys=True))
