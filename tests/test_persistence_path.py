from pathlib import Path

from app.persistence import _resolve_db_path


def test_explicit_database_path_has_highest_priority(tmp_path):
    configured = tmp_path / "chosen.sqlite3"
    mounted = tmp_path / "volume"
    mounted.mkdir()
    result = _resolve_db_path(
        {"FOUNDRY_DB_PATH": str(configured)},
        data_dir=mounted,
        project_root=tmp_path,
    )
    assert result == configured


def test_mounted_data_volume_is_preferred_over_repository_storage(tmp_path):
    mounted = tmp_path / "data-volume"
    mounted.mkdir()
    result = _resolve_db_path({}, data_dir=mounted, project_root=tmp_path)
    assert result == mounted / "strategy_foundry.sqlite3"


def test_vercel_uses_tmp_when_no_explicit_path_is_set(tmp_path):
    result = _resolve_db_path({"VERCEL": "1"}, data_dir=tmp_path, project_root=tmp_path)
    assert result == Path("/tmp/strategy_foundry.sqlite3")


def test_local_repository_fallback_when_no_volume_is_available(tmp_path):
    missing_volume = tmp_path / "not-mounted"
    result = _resolve_db_path({}, data_dir=missing_volume, project_root=tmp_path)
    assert result == tmp_path / "data" / "foundry.sqlite3"
