"""Test untuk `QADatasetLoader` (FR-2)."""

from pathlib import Path

import pytest

from rag_finetune.dataset.loader import DatasetError, QADatasetLoader

FIXTURE_PATH = "tests/fixtures/qa_sample.jsonl"


def test_loads_all_items_from_fixture():
    items = QADatasetLoader().load(FIXTURE_PATH)

    assert len(items) == 10
    assert items[0].id == "q1"
    assert items[0].golden_answers == ["Solari"]


def test_limit_truncates_without_reading_whole_file():
    items = QADatasetLoader().load(FIXTURE_PATH, limit=3)

    assert len(items) == 3
    assert [item.id for item in items] == ["q1", "q2", "q3"]


def test_malformed_json_line_raises_dataset_error_with_line_number(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "q1", "question": "a?", "golden_answers": ["x"]}\nnot json\n')

    with pytest.raises(DatasetError, match=r"broken\.jsonl:2"):
        QADatasetLoader().load(str(path))


def test_missing_required_field_raises_dataset_error_with_line_number(tmp_path: Path):
    path = tmp_path / "missing_field.jsonl"
    path.write_text('{"id": "q1", "question": "a?"}\n')

    with pytest.raises(DatasetError, match=r"missing_field\.jsonl:1"):
        QADatasetLoader().load(str(path))
