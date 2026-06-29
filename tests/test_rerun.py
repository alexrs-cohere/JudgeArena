"""Tests for rebuilding a run from its metadata via --rerun."""

from __future__ import annotations

import judgearena.repro as repro
from judgearena.config import RunConfig, build_config_from_rerun, build_run_config


def _write_metadata(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(repro, "_get_dependency_versions", lambda *a, **k: {})
    monkeypatch.setattr(repro, "_get_git_hash", lambda *a, **k: "c" * 40)
    return repro.write_run_metadata(
        output_dir=tmp_path,
        entrypoint="judgearena.test.entrypoint",
        run=cfg.model_dump(),
        config_resolved=cfg.model_dump(),
        dataset_revisions={"repo": "rev-1"},
    )


def test_rerun_roundtrips_resolved_config(tmp_path, monkeypatch):
    cfg = RunConfig(
        task="alpaca-eval",
        model={"name": "Dummy/a", "baseline": "Dummy/b", "temperature": 0.3, "seed": 7},
        judge={"model": "Dummy/j", "swap_mode": "both"},
        generation={"n_instructions": 5, "truncate_all_input_chars": 1234},
        run={"seed": 42, "result_folder": "orig-results"},
    )
    metadata_path = _write_metadata(tmp_path, cfg, monkeypatch)

    rebuilt = build_config_from_rerun(metadata_path)
    assert rebuilt.model_dump() == cfg.model_dump()


def test_rerun_allows_result_folder_override(tmp_path, monkeypatch):
    cfg = RunConfig(
        task="alpaca-eval",
        model={"name": "Dummy/a", "baseline": "Dummy/b"},
        judge={"model": "Dummy/j"},
        run={"result_folder": "orig-results"},
    )
    metadata_path = _write_metadata(tmp_path, cfg, monkeypatch)

    rebuilt = build_run_config(
        ["--rerun", str(metadata_path), "--run.result_folder", "new-results"]
    )
    assert rebuilt.run.result_folder == "new-results"
    # Result-affecting settings are still taken verbatim from the metadata.
    assert rebuilt.task == "alpaca-eval"
    assert rebuilt.model.name == "Dummy/a"


def test_rerun_via_build_run_config_dispatch(tmp_path, monkeypatch):
    cfg = RunConfig(
        task="alpaca-eval",
        model={"name": "Dummy/a", "baseline": "Dummy/b"},
        judge={"model": "Dummy/j"},
    )
    metadata_path = _write_metadata(tmp_path, cfg, monkeypatch)

    rebuilt = build_run_config(["--rerun", str(metadata_path), "-q"])
    assert rebuilt.run.verbosity == -1
    assert rebuilt.model.name == "Dummy/a"
