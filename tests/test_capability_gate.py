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
