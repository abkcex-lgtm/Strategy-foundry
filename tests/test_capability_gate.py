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
            "trades": 2, "avg_net_tp_per_day": 0.03,
            "win_rate_pct": 50.0, "profit_factor": 3.5,
            "max_consecutive_losses": 1,
        },
    }
    active_candidate = {
        "development_audit_status": "MATCH", "development_distance": 2.0,
        "development_metrics": {
            "trades": 80, "avg_net_tp_per_day": 0.5,
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
        "span_days": 10, "trades": 120, "wins": 60,
        "win_rate_pct": 50.0, "avg_net_tp_per_day": 4.0,
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
    assert distance == sum(distances.values())
