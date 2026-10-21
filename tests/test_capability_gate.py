from app.engine import _capability_evidence
from research.run_factory import near_key


def test_capability_gate_rejects_statistically_tiny_samples():
    assert _capability_evidence({
        "avg_net_tp_per_day": 0.5, "trades": 50,
        "win_rate_pct": 25.0, "profit_factor": 0.4,
    }) is True
    assert _capability_evidence({
        "avg_net_tp_per_day": 0.04, "trades": 7,
        "win_rate_pct": 57.0, "profit_factor": 4.6,
    }) is False


def test_capability_evidence_outranks_low_sample_near_miss():
    sparse_high_pf = {
        "development_audit_status": "MATCH", "development_distance": 0.98,
        "development_metrics": {
            "trades": 2, "span_days": 91, "avg_net_tp_per_day": 0.03,
            "win_rate_pct": 50.0, "profit_factor": 3.5,
            "max_consecutive_losses": 1,
        },
    }
    active_candidate = {
        "development_audit_status": "MATCH", "development_distance": 2.0,
        "development_metrics": {
            "trades": 300, "span_days": 10, "avg_net_tp_per_day": 0.5,
            "monthly_trade_counts": {"2026-07": 300},
            "win_rate_pct": 25.0, "profit_factor": 0.5,
            "max_consecutive_losses": 12,
        },
    }
    assert near_key(active_candidate) < near_key(sparse_high_pf)


def test_capability_pilot_is_advisory_and_never_blocks_requested_search():
    from research.run_factory import capability_pilot_warning

    warning = capability_pilot_warning({
        "capability_evidence_candidates": 0,
        "promising_candidates": 0,
    }, 500000)
    assert warning is not None
    assert "advisory only" in warning
    assert capability_pilot_warning({
        "capability_evidence_candidates": 2,
        "promising_candidates": 0,
    }, 500000) is None
    assert capability_pilot_warning({
        "capability_evidence_candidates": 0,
        "promising_candidates": 0,
    }, 100) is None


from app.engine import _goal_activity_evidence, _candidate_distance, normalize_contract


def test_goal_activity_evidence_rejects_sparse_sample():
    cfg = normalize_contract()
    sparse = {
        "span_days": 91, "trades": 20, "wins": 15,
        "win_rate_pct": 75.0, "avg_net_tp_per_day": 0.16,
    }
    assert _goal_activity_evidence(sparse, cfg) is False


def test_goal_activity_evidence_scales_to_target_and_observed_win_rate():
    cfg = normalize_contract()
    sufficient = {
        "span_days": 10, "trades": 300, "wins": 150,
        "win_rate_pct": 50.0, "avg_net_tp_per_day": 4.0,
        "monthly_trade_counts": {"2026-07": 300},
    }
    assert _goal_activity_evidence(sufficient, cfg) is True


def test_candidate_distance_normalizes_profit_factor_like_other_targets():
    cfg = normalize_contract()
    metrics = {
        "win_rate_pct": 30.0, "profit_factor": 0.5,
        "avg_net_tp_per_day": 1.5, "max_consecutive_losses": 16,
        "open_positions_at_end": 0,
    }
    distance, failed, distances = _candidate_distance(metrics, cfg)
    assert set(distances) == {
        "win_rate_pct", "profit_factor", "avg_net_tp_per_day",
        "max_consecutive_losses",
    }
    assert "profit_factor" in failed
    weights = {"avg_net_tp_per_day": 3.0, "trades_per_day": 2.5, "win_rate_pct": 1.5,
               "profit_factor": 1.5, "max_consecutive_losses": 1.0}
    expected = sum(weights.get(k, 1.0) * v for k, v in distances.items()) / sum(weights.get(k, 1.0) for k in distances)
    assert distance == expected
    assert "insufficient_activity_evidence" in failed



def test_goal_activity_evidence_requires_300_trades_in_each_month():
    cfg = normalize_contract()
    # 300 total trades are a floor, but the 3 TP/day goal at 50% WR needs more
    # trades over a 60-day window, so this sample is correctly insufficient.
    assert _goal_activity_evidence({
        "span_days": 60, "trades": 300, "wins": 150,
        "win_rate_pct": 50.0, "avg_net_tp_per_day": 2.5,
        "monthly_trade_counts": {"2026-07": 180, "2026-08": 120},
    }, cfg) is False
    # A single-month sample with 300 trades and enough TP/day is eligible.
    assert _goal_activity_evidence({
        "span_days": 10, "trades": 300, "wins": 150,
        "win_rate_pct": 50.0, "avg_net_tp_per_day": 15.0,
        "monthly_trade_counts": {"2026-07": 300},
    }, cfg) is True
    assert _goal_activity_evidence({
        "span_days": 90, "trades": 600, "wins": 300,
        "win_rate_pct": 50.0, "avg_net_tp_per_day": 3.33,
        "monthly_trade_counts": {"2026-04": 50, "2026-05": 100, "2026-06": 450},
    }, cfg) is False
    assert _goal_activity_evidence({
        "span_days": 90, "trades": 720, "wins": 360,
        "win_rate_pct": 50.0, "avg_net_tp_per_day": 4.0,
        "monthly_trade_counts": {"2026-04": 200, "2026-05": 250, "2026-06": 270},
    }, cfg) is False
    assert _goal_activity_evidence({
        "span_days": 90, "trades": 900, "wins": 450,
        "win_rate_pct": 50.0, "avg_net_tp_per_day": 5.0,
        "monthly_trade_counts": {"2026-04": 300, "2026-05": 300, "2026-06": 300},
    }, cfg) is True



def test_near_key_prioritizes_300_trade_train_samples_over_sparse_lucky_samples():
    from research.run_factory import near_key

    high_sample = {
        "development_audit_status": "MATCH",
        "development_distance": 4.0,
        "train_metrics": {
            "trades": 820, "span_days": 91, "wins": 340,
            "win_rate_pct": 41.5, "profit_factor": 0.9,
            "avg_net_tp_per_day": 3.74,
        },
        "development_metrics": {
            "trades": 310, "span_days": 31, "wins": 130,
            "win_rate_pct": 41.9, "profit_factor": 0.9,
            "avg_net_tp_per_day": 4.19,
        },
    }
    sparse_lucky = {
        "development_audit_status": "MATCH",
        "development_distance": 0.01,
        "train_metrics": {
            "trades": 42, "span_days": 91, "wins": 30,
            "win_rate_pct": 71.4, "profit_factor": 3.0,
            "avg_net_tp_per_day": 0.33,
        },
        "development_metrics": {
            "trades": 12, "span_days": 31, "wins": 9,
            "win_rate_pct": 75.0, "profit_factor": 3.0,
            "avg_net_tp_per_day": 0.29,
        },
    }
    assert near_key(high_sample) < near_key(sparse_lucky)
