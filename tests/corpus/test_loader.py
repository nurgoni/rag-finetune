"""Test untuk `CorpusLoader` (FR-3.1)."""

import inspect
from pathlib import Path

import pytest

from rag_finetune.corpus.loader import CorpusError, CorpusLoader

FIXTURE_PATH = "tests/fixtures/corpus_sample.jsonl"


def test_load_returns_a_generator_not_a_list():
    result = CorpusLoader().load(FIXTURE_PATH)

    assert inspect.isgenerator(result)


def test_streams_all_documents_from_fixture_with_contents_mapped_to_text():
    documents = list(CorpusLoader().load(FIXTURE_PATH))

    assert len(documents) == 20
    assert documents[0].id == "d1"
    assert documents[0].title == "Kota Arannis"
    assert "Velyria" in documents[0].text


def test_malformed_json_line_raises_corpus_error_with_line_number(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "d1", "contents": "isi"}\nnot json\n')

    with pytest.raises(CorpusError, match=r"broken\.jsonl:2"):
        list(CorpusLoader().load(str(path)))


def test_missing_contents_field_raises_corpus_error_with_line_number(tmp_path: Path):
    path = tmp_path / "missing_field.jsonl"
    path.write_text('{"id": "d1"}\n')

    with pytest.raises(CorpusError, match=r"missing_field\.jsonl:1"):
        list(CorpusLoader().load(str(path)))
