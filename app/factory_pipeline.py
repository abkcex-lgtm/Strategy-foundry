"""Train -> development -> untouched OOS validation pipeline.

This module deliberately never ranks or tunes candidates using OOS metrics. OOS is
reported only after candidate selection on Train/Development, to avoid test leakage.
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from .data import split_reference_period
from .engine import backtest, search_candidates


class FactoryPreflightError(ValueError):
    """Raised when the reference split cannot safely support a factory run."""


def _metric_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy": result["strategy"],
        "metrics": result["metrics"],
        "run_hash_sha256": result["run_hash_sha256"],
        "trade_count": len(result.get("ledger", [])),
    }


def run_validation_pipeline(
    frame: pd.DataFrame,
    candidates: int = 500,
    seed: int = 42,
    contract: dict[str, Any] | None = None,
    minimum_rows_per_split: int = 100,
    search_fn: Callable[..., dict[str, Any]] = search_candidates,
    backtest_fn: Callable[..., dict[str, Any]] = backtest,
) -> dict[str, Any]:
    """Search only Train, select on Development, and report untouched OOS once.

    The candidate engine currently returns all Train-pass candidates or exactly
    five Train near-misses. This pipeline does not pretend that this starter
    search is the full 500k+ strategy factory.
    """
    if not 1 <= int(candidates) <= 500_000:
        raise FactoryPreflightError("candidate count must be between 1 and 500000")

    splits = split_reference_period(frame)
    counts = {name: int(len(part)) for name, part in splits.items()}
    missing = [name for name, count in counts.items() if count < minimum_rows_per_split]
    if missing:
        raise FactoryPreflightError(
            "PREFLIGHT FAILED / SEARCH NOT STARTED: insufficient rows in "
            + ", ".join(missing)
            + f"; counts={counts}; minimum={minimum_rows_per_split}"
        )

    train_result = search_fn(
        splits["train"], max_candidates=int(candidates), seed=int(seed), contract=contract
    )
    candidates_found = train_result.get("results", [])
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates_found:
        spec = candidate.get("strategy")
        if not isinstance(spec, dict):
            continue
        try:
            dev = backtest_fn(splits["development"], spec, contract)
            dev_metrics = dev["metrics"]
            evaluated.append({
                "strategy": spec,
                "train_metrics": candidate.get("metrics", {}),
                "train_distance": candidate.get("distance"),
                "development": _metric_summary(dev),
                "development_pass": bool(dev_metrics.get("passes_all_targets")),
            })
        except Exception as exc:
            evaluated.append({
                "strategy": spec,
                "train_metrics": candidate.get("metrics", {}),
                "train_distance": candidate.get("distance"),
                "development": {"status": "PREFLIGHT_FAILED", "reason": str(exc)},
                "development_pass": False,
            })

    # Ranking and candidate selection use Train + Development only.
    def rank(item: dict[str, Any]) -> tuple[Any, ...]:
        m = item.get("development", {}).get("metrics", {})
        return (
            not item.get("development_pass", False),
            float(item.get("train_distance") or 0),
            int(m.get("max_consecutive_losses", 10**9)),
            -float(m.get("win_rate_pct", 0)),
            -float(m.get("avg_net_tp_per_day", 0)),
            -float(m.get("profit_factor") or 0),
        )

    evaluated.sort(key=rank)
    train_and_dev_pass = [item for item in evaluated if item["development_pass"]]
    selected_for_oos = train_and_dev_pass if train_and_dev_pass else evaluated[:5]

    # OOS is a final report only. It is not used to reorder, tune, or select.
    final_results = []
    for item in selected_for_oos:
        try:
            oos = backtest_fn(splits["oos"], item["strategy"], contract)
            item["oos"] = _metric_summary(oos)
            item["oos_pass"] = bool(oos["metrics"].get("passes_all_targets"))
            item["passes_train_development_oos"] = bool(
                item["development_pass"]
                and item["train_metrics"].get("passes_all_targets", False)
                and item["oos_pass"]
            )
        except Exception as exc:
            item["oos"] = {"status": "PREFLIGHT_FAILED", "reason": str(exc)}
            item["oos_pass"] = False
            item["passes_train_development_oos"] = False
        final_results.append(item)

    # Preserve development ordering; OOS results must not influence ranking.
    passing = [item for item in final_results if item["passes_train_development_oos"]]
    output = passing if passing else final_results[:5]
    return {
        "status": "COMPLETED",
        "search_started": True,
        "requested_candidates": int(candidates),
        "evaluated_train_candidates": int(train_result.get("evaluated_candidates", 0)),
        "failed_train_candidates": int(train_result.get("failed_candidates", 0)),
        "seed": int(seed),
        "contract": train_result.get("contract", contract or {}),
        "dataset_rows": len(frame),
        "split_rows": counts,
        "periods_utc": {
            "train": "2026-04-01 <= timestamp < 2026-07-01",
            "development": "2026-07-01 <= timestamp < 2026-08-01",
            "oos": "2026-08-01 <= timestamp < 2026-10-01",
        },
        "selection_policy": "Train search; Development ranking; OOS final report only; never rank by OOS",
        "all_targets_passed": bool(passing),
        "results": output,
        "limitations": [
            "Search family and candidate count remain bounded by the current EMA-only starter engine.",
            "This pipeline is not a profitability certificate and does not replace independent trade-by-trade audit.",
            "The OOS set must remain untouched after this report; further tuning invalidates its independence.",
        ],
    }
