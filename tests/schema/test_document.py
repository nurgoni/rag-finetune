"""Test untuk Document dan RetrievedDocument (PRD §6.3)."""

from rag_finetune.schema import Document, RetrievedDocument


def test_document_defaults_title_and_metadata():
    doc = Document(id="d1", text="isi dokumen")

    assert doc.title is None
    assert doc.metadata == {}


def test_document_is_frozen():
    doc = Document(id="d1", text="isi dokumen")

    try:
        doc.text = "diubah"
    except Exception as exc:  # noqa: BLE001 - memverifikasi bahwa pydantic menolak mutasi
        assert "frozen" in str(exc).lower()
    else:
        raise AssertionError("Document seharusnya immutable")


def test_retrieved_document_carries_score_and_rank():
    doc = Document(id="d1", text="isi dokumen")
    retrieved = RetrievedDocument(document=doc, score=0.73, rank=0)

    assert retrieved.document == doc
    assert retrieved.score == 0.73
    assert retrieved.rank == 0
