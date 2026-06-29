"""Tests for canonical run descriptors and content-addressed cache keys."""

from __future__ import annotations

import judgearena.descriptor as descriptor
from judgearena.config import RunConfig
from judgearena.descriptor import (
    build_run_descriptor,
    completion_descriptor,
    descriptor_hash,
    judge_descriptor,
)
from judgearena.prompts.registry import resolve_run_judge_prompt


def _pairwise_cfg(**overrides):
    model = {"name": "Dummy/a", "baseline": "Dummy/b"}
    judge = {"model": "Dummy/j"}
    model.update(overrides.pop("model", {}))
    judge.update(overrides.pop("judge", {}))
    return RunConfig(task="alpaca-eval", model=model, judge=judge, **overrides)


def test_descriptor_hash_is_order_independent_and_stable():
    a = {"x": 1, "y": [1, 2], "z": {"b": 2, "a": 1}}
    b = {"z": {"a": 1, "b": 2}, "y": [1, 2], "x": 1}
    assert descriptor_hash(a) == descriptor_hash(b)
    # Full-length digest when length=None, truncated otherwise.
    assert len(descriptor_hash(a, length=None)) == 64
    assert len(descriptor_hash(a)) == 16


def test_completion_descriptor_busts_on_dataset_revision_change():
    cfg = _pairwise_cfg()
    gen = cfg.model.evaluated_generation_kwargs()
    base = completion_descriptor(
        cfg, "Dummy/a", generation_kwargs=gen, dataset_revisions={"repo": "rev-1"}
    )
    bumped = completion_descriptor(
        cfg, "Dummy/a", generation_kwargs=gen, dataset_revisions={"repo": "rev-2"}
    )
    assert descriptor_hash(base) != descriptor_hash(bumped)


def test_completion_descriptor_busts_on_sampling_change():
    cfg_a = _pairwise_cfg(model={"temperature": 0.0})
    cfg_b = _pairwise_cfg(model={"temperature": 1.0})
    desc_a = completion_descriptor(
        cfg_a, "Dummy/a", generation_kwargs=cfg_a.model.evaluated_generation_kwargs()
    )
    desc_b = completion_descriptor(
        cfg_b, "Dummy/a", generation_kwargs=cfg_b.model.evaluated_generation_kwargs()
    )
    assert descriptor_hash(desc_a) != descriptor_hash(desc_b)


def test_judge_descriptor_busts_on_judge_fields():
    cfg = _pairwise_cfg()
    prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)

    def make(cfg_):
        prompt_ = resolve_run_judge_prompt(cfg_.task, cfg_.judge)
        return descriptor_hash(
            judge_descriptor(
                cfg_,
                resolved_prompt=prompt_,
                judge_kwargs=cfg_.judge.model_kwargs(),
                completion_keys=["abc"],
            )
        )

    base = make(cfg)
    assert base == descriptor_hash(
        judge_descriptor(
            cfg,
            resolved_prompt=prompt,
            judge_kwargs=cfg.judge.model_kwargs(),
            completion_keys=["abc"],
        )
    )
    assert base != make(_pairwise_cfg(judge={"model": "Dummy/other"}))
    assert base != make(_pairwise_cfg(judge={"temperature": 0.7}))
    assert base != make(_pairwise_cfg(judge={"swap_mode": "both"}))
    assert base != make(_pairwise_cfg(judge={"provide_explanation": True}))


def test_judge_descriptor_busts_when_completion_keys_change():
    cfg = _pairwise_cfg()
    prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)
    kw = cfg.judge.model_kwargs()
    a = descriptor_hash(
        judge_descriptor(
            cfg, resolved_prompt=prompt, judge_kwargs=kw, completion_keys=["k1"]
        )
    )
    b = descriptor_hash(
        judge_descriptor(
            cfg, resolved_prompt=prompt, judge_kwargs=kw, completion_keys=["k2"]
        )
    )
    assert a != b


def test_run_descriptor_ignores_non_result_fields():
    base = build_run_descriptor(_pairwise_cfg())
    other = build_run_descriptor(
        _pairwise_cfg(run={"result_folder": "somewhere-else", "verbosity": 3})
    )
    assert descriptor_hash(base) == descriptor_hash(other)


def test_run_descriptor_tracks_result_affecting_fields():
    base = build_run_descriptor(_pairwise_cfg())
    changed_seed = build_run_descriptor(_pairwise_cfg(run={"seed": 99}))
    changed_trunc = build_run_descriptor(
        _pairwise_cfg(generation={"truncate_all_input_chars": 123})
    )
    assert descriptor_hash(base) != descriptor_hash(changed_seed)
    assert descriptor_hash(base) != descriptor_hash(changed_trunc)


def test_resolve_dataset_revisions_for_elo(monkeypatch):
    monkeypatch.setattr(descriptor, "hf_revision", lambda repo_id: f"rev::{repo_id}")
    cfg = RunConfig(
        task="elo-comparia",
        model={"name": "Dummy/a"},
        judge={"model": "Dummy/j"},
    )
    revisions = descriptor.resolve_dataset_revisions(cfg)
    assert revisions == {
        "ministere-culture/comparia-votes": "rev::ministere-culture/comparia-votes"
    }
