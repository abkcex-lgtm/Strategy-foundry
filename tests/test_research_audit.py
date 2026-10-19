from copy import deepcopy

from app.audit import FIELDS
from research.run_factory import ledger_match, near_key, trade_diagnostics


def _trade(trade_id, exit_reason, net_pnl, streak, entry_time, exit_time, balance):
    row = {field: 0 for field in FIELDS}
    row.update({
        "trade_id": trade_id,
        "signal_time": entry_time - 60_000,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "direction": "LONG",
        "signal_price": 100.0,
        "entry_price": 100.0,
        "stop_loss": 99.0,
        "take_profit": 103.5,
        "exit_price": 103.5 if exit_reason == "TP" else 99.0,
        "exit_reason": exit_reason,
        "quantity": 1.0,
        "risk_budget": 10.0,
        "gross_pnl": net_pnl + 2.0,
        "entry_cost": 1.0,
        "exit_cost": 1.0,
        "fees_slippage": 2.0,
        "net_pnl": net_pnl,
        "balance_after": balance,
        "loss_streak_after": streak,
        "resolution_status": "resolved_base_ohlc",
        "execution_ambiguity": False,
    })
    return row


def _primary_and_independent():
    ledger = [
        _trade(1, "TP", 35.0, 0, 60_000, 120_000, 1035.0),
        _trade(2, "SL", -10.0, 1, 180_000, 240_000, 1025.0),
    ]
    primary = {
        "ledger": ledger,
        "open_position": None,
        "contract": {"initial_balance": 1000.0},
        "metrics": {
            "trades": 2, "wins": 1, "losses": 1, "win_rate_pct": 50.0,
            "profit_factor": 3.5, "avg_net_tp_per_day": 0.5,
            "max_consecutive_losses": 1, "net_pnl": 25.0,
            "final_balance": 1025.0, "total_costs": 4.0,
            "unresolved_as_loss_count": 0, "open_positions_at_end": 0,
            "open_position_unrealized_net_pnl": 0.0, "span_days": 2.0,
        },
    }
    independent = {
        "ledger": deepcopy(ledger), "open_position": None,
        "max_consecutive_losses": 1,
    }
    return primary, independent


def test_ledger_audit_checks_aggregate_metrics_not_only_trade_rows():
    primary, independent = _primary_and_independent()
    matched, evidence = ledger_match(primary, independent)
    assert matched is True
    assert "profit_factor" in evidence["aggregate_metrics_checked"]

    wrong = deepcopy(primary)
    wrong["metrics"]["profit_factor"] = 0.26
    matched, evidence = ledger_match(wrong, independent)
    assert matched is False
    assert evidence["mismatch"] == "aggregate_metric"
    assert evidence["field"] == "profit_factor"


def test_near_miss_ranking_uses_full_criterion_distance_before_frequency():
    more_frequent_but_worse = {
        "development_audit_status": "MATCH",
        "development_distance": 0.25,
        "development_metrics": {
            "avg_net_tp_per_day": 2.9, "profit_factor": 0.7,
            "win_rate_pct": 30.0, "max_consecutive_losses": 20,
        },
    }
    less_frequent_but_closer = {
        "development_audit_status": "MATCH",
        "development_distance": 0.10,
        "development_metrics": {
            "avg_net_tp_per_day": 1.5, "profit_factor": 0.9,
            "win_rate_pct": 38.0, "max_consecutive_losses": 12,
        },
    }
    assert near_key(less_frequent_but_closer) < near_key(more_frequent_but_worse)


def test_trade_diagnostics_expose_exit_reasons_and_loss_streak_distribution():
    ledger = [
        _trade(1, "SL", -10.0, 1, 60_000, 120_000, 990.0),
        _trade(2, "SL", -10.0, 2, 180_000, 240_000, 980.0),
        _trade(3, "TP", 35.0, 0, 300_000, 360_000, 1015.0),
        _trade(4, "SL", -10.0, 1, 420_000, 480_000, 1005.0),
    ]
    diag = trade_diagnostics(ledger, span_days=1.0)
    assert diag["exit_reason_counts"] == {"SL": 3, "TP": 1}
    assert diag["max_consecutive_losses"] == 2
    assert diag["loss_streak_run_count"] == 2
    assert diag["loss_streak_length_histogram"] == {"1": 1, "2": 1}
    assert diag["direction_breakdown"]["LONG"]["trades"] == 4
    assert diag["holding_minutes"]["max"] == 1.0
