"""Model data untuk dokumen corpus dan hasil retrieval (PRD §6.3)."""

from pydantic import BaseModel, Field


class Document(BaseModel, frozen=True):
    """Satu dokumen dari corpus.

    Args:
        id: Pengenal unik dokumen di dalam corpus.
        text: Isi dokumen.
        title: Judul dokumen. `None` kalau corpus tidak menyediakannya.
        metadata: Informasi tambahan bebas, misal sumber atau tanggal.
    """

    id: str
    text: str
    title: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedDocument(BaseModel, frozen=True):
    """Satu dokumen hasil pencarian retriever, beserta peringkatnya.

    Args:
        document: Dokumen yang ditemukan.
        score: Skor kemiripan dari retriever. Satuannya tergantung retriever
            (BM25 dan cosine similarity dense punya skala berbeda).
        rank: Peringkat dokumen ini di antara hasil pencarian yang sama,
            dimulai dari 0 untuk hasil paling relevan (FR-4.3).
    """

    document: Document
    score: float
    rank: int


if __name__ == "__main__":
    doc = Document(id="d1", text="Bola lampu ditemukan oleh Thomas Edison.", title="Bola Lampu")
    retrieved = RetrievedDocument(document=doc, score=0.87, rank=0)
    print(retrieved.model_dump_json(indent=2))
