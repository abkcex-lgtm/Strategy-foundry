import gzip
import json
import zipfile
from pathlib import Path

from research.aggregate_shards import aggregate


def _candidate(name: str, distance: float) -> dict:
    return {
        "strategy": {"name": name, "fast": 5, "slow": 20, "seed": 42},
        "contract": {"symbol": "ETHUSDT"},
        "development_metrics": {"trades": 200, "win_rate_pct": 30.0, "profit_factor": 0.8, "avg_net_tp_per_day": 1.0, "max_consecutive_losses": 10},
        "development_distance": distance,
        "development_audit_status": "MATCH",
        "development_passes_all_targets": False,
        "oos_audit_status": "MATCH",
        "oos_passes_all_targets": False,
        "full_data_audit_status": "MATCH",
        "full_run_metrics": {"trades": 200},
        "full_run_hash_sha256": "f" * 64,
    }


def _make_shard(root: Path, name: str, hashes: list[str], candidates: list[dict]) -> None:
    results_dir = root / name / "research-results"
    results_dir.mkdir(parents=True)
    report = {
        "source_commit": "abc123",
        "requested_candidates": len(hashes),
        "evaluated_unique_candidates": len(hashes),
        "candidate_search_complete": True,
        "outcome": "INCOMPLETE_SEARCH_NO_CERTIFIED_STRATEGY",
        "family_evaluated_counts": {"ema_crossover": len(hashes)},
        "data": {"dataset_sha256": "d" * 64},
        "splits": {
            "train": {"sha256": "1" * 64},
            "development": {"sha256": "2" * 64},
            "oos": {"sha256": "3" * 64},
        },
        "contract": {"symbol": "ETHUSDT"},
        "target_criteria": {"win_rate_pct_strictly_greater_than": 40},
        "results": candidates,
    }
    (results_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with gzip.open(results_dir / "evaluated_genome_hashes.txt.gz", "wt", encoding="ascii") as stream:
        stream.write("\n".join(hashes) + "\n")
    with zipfile.ZipFile(results_dir / "validated_strategies.zip", "w") as archive:
        archive.writestr("market_data_manifest.json", "{}")
        for i, candidate in enumerate(candidates, 1):
            prefix = f"strategy_{i:03d}"
            archive.writestr(prefix + ".py", "# standalone test fixture")
            for suffix in ("_train_ledger.json", "_development_ledger.json", "_full_ledger.json", "_oos_ledger.json"):
                archive.writestr(prefix + suffix, "[]")
            archive.writestr(prefix + "_manifest.json", json.dumps({
                "strategy": candidate["strategy"],
                "contract": candidate["contract"],
            }))


def test_aggregate_globally_deduplicates_and_returns_exact_five_near_misses(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    output_dir = tmp_path / "out"
    artifact_root.mkdir()
    a, b, c, d, e = (_candidate(name, distance) for name, distance in [
        ("a", 0.1), ("b", 0.2), ("c", 0.3), ("d", 0.4), ("e", 0.5)
    ])
    c_worse_duplicate = _candidate("c", 0.9)
    _make_shard(artifact_root, "worker-0", ["h1", "h2", "h3"], [a, b, c])
    _make_shard(artifact_root, "worker-1", ["h3", "h4", "h5"], [c_worse_duplicate, d, e])

    result = aggregate(artifact_root, output_dir, target=5)

    assert result["evaluated_unique_candidates"] == 5
    assert result["total_worker_evaluations_before_cross_worker_deduplication"] == 6
    assert result["duplicate_genomes_across_workers"] == 1
    assert result["candidate_search_complete"] is True
    assert result["validated_strategy_count"] == 0
    assert result["returned_candidate_count"] == 5
    assert [item["strategy"]["name"] for item in result["results"]] == ["a", "b", "c", "d", "e"]
    with zipfile.ZipFile(output_dir / "validated_strategies.zip") as archive:
        assert len([name for name in archive.namelist() if name.endswith(".py")]) == 5
        assert len([name for name in archive.namelist() if name.endswith("_manifest.json")]) == 5


def test_aggregate_rejects_mismatched_data_splits(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    candidates = [_candidate("a", 0.1), _candidate("b", 0.2), _candidate("c", 0.3)]
    _make_shard(artifact_root, "worker-0", ["h1", "h2", "h3"], candidates)
    _make_shard(artifact_root, "worker-1", ["h4", "h5", "h6"], candidates)
    second_report = artifact_root / "worker-1" / "research-results" / "report.json"
    report = json.loads(second_report.read_text())
    report["splits"]["oos"]["sha256"] = "x" * 64
    second_report.write_text(json.dumps(report))

    import pytest
    with pytest.raises(RuntimeError, match="disagree on oos_sha256"):
        aggregate(artifact_root, tmp_path / "out", target=5)
