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


def test_legacy_database_is_safely_migrated_to_empty_mounted_volume(tmp_path):
    import sqlite3
    from app.persistence import _choose_db_path

    root = tmp_path / "project"
    legacy_dir = root / "data"
    legacy_dir.mkdir(parents=True)
    legacy = legacy_dir / "foundry.sqlite3"
    with sqlite3.connect(legacy) as connection:
        connection.execute("CREATE TABLE migration_probe (value TEXT NOT NULL)")
        connection.execute("INSERT INTO migration_probe(value) VALUES ('preserve-me')")

    mounted = tmp_path / "mounted-volume"
    mounted.mkdir()
    selected, status = _choose_db_path({}, data_dir=mounted, project_root=root)

    assert selected == mounted / "strategy_foundry.sqlite3"
    assert status == "migrated_legacy_database"
    with sqlite3.connect(selected) as connection:
        assert connection.execute("SELECT value FROM migration_probe").fetchone()[0] == "preserve-me"
    assert legacy.exists()  # migration is copy-first; never delete the source


def test_existing_volume_database_is_never_overwritten_by_legacy_file(tmp_path):
    import sqlite3
    from app.persistence import _choose_db_path

    root = tmp_path / "project"
    legacy_dir = root / "data"
    legacy_dir.mkdir(parents=True)
    with sqlite3.connect(legacy_dir / "foundry.sqlite3") as connection:
        connection.execute("CREATE TABLE legacy_data (value TEXT)")
        connection.execute("INSERT INTO legacy_data VALUES ('old')")

    mounted = tmp_path / "mounted-volume"
    mounted.mkdir()
    destination = mounted / "strategy_foundry.sqlite3"
    with sqlite3.connect(destination) as connection:
        connection.execute("CREATE TABLE volume_data (value TEXT)")
        connection.execute("INSERT INTO volume_data VALUES ('new')")

    selected, status = _choose_db_path({}, data_dir=mounted, project_root=root)
    assert selected == destination
    assert status == "existing_volume_database"
    with sqlite3.connect(selected) as connection:
        assert connection.execute("SELECT value FROM volume_data").fetchone()[0] == "new"
