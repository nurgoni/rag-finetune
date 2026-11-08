"""Test untuk `read_jsonl` dan `write_jsonl` (FR-2.3)."""

from pathlib import Path

import pytest

from rag_finetune.utils.jsonl import JsonlDecodeError, read_jsonl, write_jsonl


def test_write_then_read_round_trips(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    write_jsonl(str(path), [{"id": "1"}, {"id": "2"}])

    rows = [row for _, row in read_jsonl(str(path))]

    assert rows == [{"id": "1"}, {"id": "2"}]


def test_read_jsonl_skips_blank_lines_without_shifting_line_numbers(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": "1"}\n\n{"id": "2"}\n')

    line_numbers = [line_number for line_number, _ in read_jsonl(str(path))]

    assert line_numbers == [1, 3]


def test_read_jsonl_reports_physical_line_number_on_malformed_json(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": "1"}\nnot valid json\n')

    with pytest.raises(JsonlDecodeError, match="data.jsonl:2") as exc_info:
        list(read_jsonl(str(path)))

    assert exc_info.value.line_number == 2


def test_write_jsonl_creates_parent_directory(tmp_path: Path):
    path = tmp_path / "nested" / "data.jsonl"
    write_jsonl(str(path), [{"id": "1"}])

    assert path.exists()
