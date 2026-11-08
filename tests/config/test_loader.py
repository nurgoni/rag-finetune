"""Test untuk `ConfigLoader` dan `deep_merge` (FR-1.1, FR-1.2, FR-1.4)."""

from pathlib import Path

import pytest

from rag_finetune.config.loader import ConfigError, ConfigLoader, deep_merge


def test_deep_merge_overrides_nested_keys_without_dropping_siblings():
    base = {"retriever": {"type": "random", "top_k": 5}, "run": {"seed": 42}}
    override = {"retriever": {"top_k": 10}}

    merged = deep_merge(base, override)

    assert merged == {"retriever": {"type": "random", "top_k": 10}, "run": {"seed": 42}}


def test_deep_merge_replaces_lists_instead_of_concatenating():
    base = {"evaluation": {"metrics": ["exact_match", "f1"]}}
    override = {"evaluation": {"metrics": ["exact_match"]}}

    merged = deep_merge(base, override)

    assert merged == {"evaluation": {"metrics": ["exact_match"]}}


def test_loader_merges_base_and_experiment_then_validates(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text("retriever:\n  type: random\n  top_k: 5\n")
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        "dataset:\n  path: d.jsonl\ncorpus:\n  path: c.jsonl\nretriever:\n  top_k: 3\n"
    )

    config = ConfigLoader().load(str(base_path), str(experiment_path))

    assert config.dataset.path == "d.jsonl"
    assert config.retriever.type == "random"
    assert config.retriever.top_k == 3


def test_loader_raises_config_error_with_field_path_on_wrong_type(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        "dataset:\n  path: d.jsonl\ncorpus:\n  path: c.jsonl\nretriever:\n  top_k: not_a_number\n"
    )

    with pytest.raises(ConfigError, match="retriever.top_k"):
        ConfigLoader().load(str(base_path))


def test_loader_raises_config_error_with_field_path_on_unknown_field(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        "dataset:\n  path: d.jsonl\ncorpus:\n  path: c.jsonl\nretriever:\n  bogus_field: 1\n"
    )

    with pytest.raises(ConfigError, match="retriever") as exc_info:
        ConfigLoader().load(str(base_path))

    assert "bogus_field" in str(exc_info.value)


def test_loader_treats_empty_yaml_as_empty_dict(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text("")
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("dataset:\n  path: d.jsonl\ncorpus:\n  path: c.jsonl\n")

    config = ConfigLoader().load(str(base_path), str(experiment_path))

    assert config.dataset.path == "d.jsonl"
