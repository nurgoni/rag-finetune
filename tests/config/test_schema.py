"""Test untuk `Config` dan sub-model-nya (FR-1, ADR-016)."""

import pytest
from pydantic import ValidationError

from rag_finetune.config.schema import Config, CorpusConfig, DatasetConfig


def _config(**overrides: object) -> Config:
    defaults = {
        "dataset": DatasetConfig(path="d.jsonl"),
        "corpus": CorpusConfig(path="c.jsonl"),
    }
    defaults.update(overrides)
    return Config(**defaults)  # type: ignore[arg-type]


def test_config_applies_sub_model_defaults():
    config = _config()

    assert config.retriever.type == "random"
    assert config.run.seed == 42
    assert config.run.pipeline == "sequential"


def test_config_rejects_unknown_field():
    with pytest.raises(ValidationError, match="retriever"):
        Config(
            dataset=DatasetConfig(path="d.jsonl"),
            corpus=CorpusConfig(path="c.jsonl"),
            retriever={"type": "random", "unknown_field": 1},
        )


def test_config_hash_is_stable_for_identical_config():
    first = _config()
    second = _config()

    assert first.config_hash() == second.config_hash()


def test_config_hash_differs_when_any_field_differs():
    baseline = _config()
    changed = _config(dataset=DatasetConfig(path="d.jsonl", limit=10))

    assert baseline.config_hash() != changed.config_hash()


def test_config_hash_differs_for_nested_pipeline_param():
    baseline = _config()
    changed = _config(run={"seed": 42, "pipeline": "loop"})

    assert baseline.config_hash() != changed.config_hash()
