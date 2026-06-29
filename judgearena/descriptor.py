"""Canonical run descriptors and content-addressed cache keys.

A *descriptor* is a JSON-serialisable dict capturing every setting that can
change the bytes of a produced artifact (model completions, judge outputs) or
the final results.  Hashing a descriptor yields a stable cache key, so changing
any result-affecting knob (sampling params, dataset revision, judge prompt, ...)
busts the relevant cache instead of silently reusing a stale run.

Two layers:

* **Artifact descriptors** (:func:`completion_descriptor` / :func:`judge_descriptor`)
  drive the on-disk completion/judge caches.  They contain only the fields that
  change that artifact's bytes.
* **The run descriptor** (:func:`build_run_descriptor`) is the superset recorded
  in the run metadata and consumed by ``--rerun``.  It also captures fields that
  affect the final numbers but not the cached artifacts (e.g. ELO
  soft-temperature, number of bootstraps) and intentionally excludes non-result
  fields (output folder, verbosity, caching toggle, ...).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from judgearena.dataset_revisions import RAW_URL_REVISIONS, hf_revision
from judgearena.repro import _stable_json_dumps

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.prompts.registry import ResolvedJudgePrompt

RUN_DESCRIPTOR_VERSION = "judgearena-run-descriptor/v1"

_RAW_PREFIX = "raw:"


def descriptor_hash(value: Any, *, length: int | None = 16) -> str:
    """Stable, order-independent SHA-256 hash of a JSON-serialisable descriptor.

    ``length`` truncates the hex digest (16 keeps cache names short while staying
    collision-safe in practice); pass ``None`` to keep the full digest, e.g. for
    provenance fields recorded in the run metadata.
    """
    digest = hashlib.sha256(_stable_json_dumps(value).encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def safe_name(value: str) -> str:
    """Sanitise a model/arena identifier for use inside a cache file name."""
    return value.replace("/", "_")


def _relevant_repo_ids(cfg: RunConfig) -> list[str]:
    """Best-effort mapping from a run config to the dataset repo ids it reads."""
    if cfg.elo is not None:
        # Reuse the loader's canonical arena -> repo-id mapping.
        from judgearena.arenas_utils import arena_repo_ids

        return arena_repo_ids(cfg.elo.arena)

    task = cfg.task
    if task == "mt-bench":
        return ["lmsys/mt-bench", f"{_RAW_PREFIX}lm-sys/FastChat"]
    if task.startswith("fluency"):
        return ["geoalgo/multilingual-contexts-to-be-completed"]
    if task == "alpaca-eval":
        return ["judge-arena/judge-arena-dataset"]

    from judgearena.instruction_dataset.m_arenahard import (
        _M_ARENA_HARD_HF_REPOS,
        split_m_arena_hard_dataset,
    )

    parsed = split_m_arena_hard_dataset(task)
    if parsed is not None:
        version_key, _ = parsed
        repo = _M_ARENA_HARD_HF_REPOS.get(version_key)
        return [repo] if repo else []

    from judgearena.instruction_dataset.arena_hard import (
        ARENA_HARD_HF_REPO_ID,
        is_arena_hard_dataset,
    )

    if is_arena_hard_dataset(task):
        return [ARENA_HARD_HF_REPO_ID]
    return []


def resolve_dataset_revisions(cfg: RunConfig) -> dict[str, str | None]:
    """Resolve ``{repo_id: pinned_revision}`` for the datasets a run reads.

    Folding this into the completion cache key closes the gap where bumping a
    pinned dataset revision did not invalidate cached completions.  Unknown or
    unpinned repos are recorded with a ``None`` revision so the gap stays
    visible in the metadata.
    """
    revisions: dict[str, str | None] = {}
    for repo_id in _relevant_repo_ids(cfg):
        if repo_id.startswith(_RAW_PREFIX):
            revisions[repo_id] = RAW_URL_REVISIONS.get(repo_id[len(_RAW_PREFIX) :])
        else:
            revisions[repo_id] = hf_revision(repo_id)
    return revisions


def completion_descriptor(
    cfg: RunConfig,
    model_spec: str,
    *,
    generation_kwargs: dict[str, Any],
    selection: dict[str, Any] | None = None,
    dataset_revisions: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Descriptor for a single model's completions on a task.

    ``generation_kwargs`` should be the *resolved* kwargs actually passed to the
    backend (including any derived values such as a thinking-token sub-budget) so
    the cache key matches the bytes that were generated.  ``selection`` carries
    any non-``head`` instruction-selection metadata (used by the ELO path for
    seeded-random sampling / language filtering).
    """
    return {
        "descriptor_version": RUN_DESCRIPTOR_VERSION,
        "kind": "completion",
        "task": cfg.task,
        "model_spec": model_spec,
        "n_instructions": cfg.generation.n_instructions,
        "truncate_all_input_chars": cfg.generation.truncate_all_input_chars,
        "generation_kwargs": generation_kwargs,
        "dataset_revisions": (
            dataset_revisions
            if dataset_revisions is not None
            else resolve_dataset_revisions(cfg)
        ),
        "selection": selection,
    }


def judge_descriptor(
    cfg: RunConfig,
    *,
    resolved_prompt: ResolvedJudgePrompt,
    judge_kwargs: dict[str, Any],
    completion_keys: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Descriptor for a judge pass over a set of (cached) completions.

    ``completion_keys`` are the hashes of the completion descriptors the judge
    scores, so any change to the underlying completions also busts the judge
    cache.  ``extra`` carries path-specific knobs (e.g. the ELO ``run_seed`` that
    drives opponent/position sampling).
    """
    descriptor: dict[str, Any] = {
        "descriptor_version": RUN_DESCRIPTOR_VERSION,
        "kind": "judge",
        "task": cfg.task,
        "judge_model": cfg.judge.model,
        "judge_kwargs": judge_kwargs,
        "swap_mode": cfg.judge.swap_mode,
        "provide_explanation": cfg.judge.provide_explanation,
        "strip_thinking_before_judging": cfg.judge.strip_thinking_before_judging,
        "battle_thinking_token_budget": cfg.judge.battle_thinking_token_budget,
        "truncate_judge_input_chars": cfg.generation.truncate_judge_input_chars,
        "prompt_preset": getattr(resolved_prompt, "preset_name", None),
        "parser_mode": getattr(resolved_prompt, "parser_mode", None),
        "prompt_system_sha256": getattr(resolved_prompt, "system_sha256", None),
        "prompt_user_sha256": getattr(resolved_prompt, "user_sha256", None),
        "prompt_delegated": getattr(resolved_prompt, "delegated", False),
        "completion_keys": list(completion_keys),
    }
    if extra:
        descriptor.update(extra)
    return descriptor


def build_run_descriptor(cfg: RunConfig) -> dict[str, Any]:
    """Build the canonical, result-affecting descriptor for a whole run.

    This is the single source of truth recorded in the run metadata and replayed
    by ``--rerun``.  It excludes purely operational settings (``result_folder``,
    ``verbosity``, ``log_file``, ``use_tqdm``, ``ignore_cache``).
    """
    descriptor: dict[str, Any] = {
        "descriptor_version": RUN_DESCRIPTOR_VERSION,
        "task": cfg.task,
        "run_seed": cfg.run.seed,
        "dataset_revisions": resolve_dataset_revisions(cfg),
        "generation": {
            "n_instructions": cfg.generation.n_instructions,
            "truncate_all_input_chars": cfg.generation.truncate_all_input_chars,
            "truncate_judge_input_chars": cfg.generation.truncate_judge_input_chars,
        },
        "model": {
            "name": cfg.model.name,
            "baseline": cfg.model.baseline,
            "evaluated_generation_kwargs": cfg.model.evaluated_generation_kwargs(),
            "baseline_generation_kwargs": (
                None if cfg.elo is not None else cfg.model.baseline_generation_kwargs()
            ),
        },
        "judge": {
            "model": cfg.judge.model,
            "model_kwargs": cfg.judge.model_kwargs(
                fallback_chat_template=cfg.model.chat_template
            ),
            "swap_mode": cfg.judge.swap_mode,
            "provide_explanation": cfg.judge.provide_explanation,
            "strip_thinking_before_judging": cfg.judge.strip_thinking_before_judging,
            "battle_thinking_token_budget": cfg.judge.battle_thinking_token_budget,
            "prompt_preset": cfg.judge.prompt_preset,
            "system_prompt_file": cfg.judge.system_prompt_file,
            "user_prompt_file": cfg.judge.user_prompt_file,
        },
    }

    if cfg.elo is not None:
        descriptor["elo"] = {
            "arena": cfg.elo.arena,
            "baseline_model": cfg.elo.baseline_model,
            "n_bootstraps": cfg.elo.n_bootstraps,
            "languages": sorted(cfg.elo.languages) if cfg.elo.languages else None,
            "n_instructions_per_language": cfg.elo.n_instructions_per_language,
            "elo_random_battles": cfg.elo.elo_random_battles,
            "soft_elo": cfg.elo.soft_elo,
            "soft_elo_temperature": cfg.elo.soft_elo_temperature,
            "calibrate_temperature": cfg.elo.calibrate_temperature,
            "calibration_size": cfg.elo.calibration_size,
        }

    # Best-effort: fold the resolved judge prompt hashes in so prompt-file edits
    # change the descriptor even though the file paths stay the same.
    try:
        from judgearena.prompts.registry import resolve_run_judge_prompt

        prompt_task = cfg.elo.arena if cfg.elo is not None else cfg.task
        resolved_prompt = resolve_run_judge_prompt(
            prompt_task, cfg.judge, multi_turn=(cfg.task == "mt-bench")
        )
        descriptor["judge"]["resolved_prompt_preset"] = resolved_prompt.preset_name
        descriptor["judge"]["prompt_system_sha256"] = resolved_prompt.system_sha256
        descriptor["judge"]["prompt_user_sha256"] = resolved_prompt.user_sha256
        descriptor["judge"]["prompt_delegated"] = resolved_prompt.delegated
    except Exception:
        pass

    return descriptor
