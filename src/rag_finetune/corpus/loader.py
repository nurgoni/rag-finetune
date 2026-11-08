"""Baca corpus dokumen dari JSONL jadi `Document`, streaming (FR-3.1)."""

from collections.abc import Iterator

from rag_finetune.exceptions import RagFinetuneError
from rag_finetune.schema import Document
from rag_finetune.utils.jsonl import JsonlDecodeError, read_jsonl


class CorpusError(RagFinetuneError):
    """Corpus gagal dibaca atau divalidasi."""


class CorpusLoader:
    """Memuat corpus dari JSONL jadi `Iterator[Document]`, tanpa memuat semuanya ke memori.

    Baris JSONL memakai konvensi FlashRAG: field `contents` (bukan `text`) dan
    `title` opsional (FR-3.1, ADR-018). Loader memetakan `contents` ke
    `Document.text` supaya corpus HotpotQA asli bisa dipakai nanti tanpa
    konversi field tambahan.
    """

    def load(self, path: str) -> Iterator[Document]:
        """Baca JSONL corpus baris demi baris.

        Args:
            path: Path ke file JSONL corpus.

        Yields:
            Satu `Document` per baris, urut sesuai urutan baris di file.
            Iterasi ini streaming — corpus besar tidak perlu masuk memori
            sekaligus.

        Raises:
            CorpusError: Kalau sebuah baris bukan JSON valid, atau tidak
                punya field `id`/`contents` yang dibutuhkan. Pesan menyebut
                path file dan nomor baris (semangat FR-2.3 untuk corpus).
        """
        try:
            for line_number, row in read_jsonl(path):
                try:
                    yield Document(id=row["id"], text=row["contents"], title=row.get("title"))
                except KeyError as exc:
                    msg = f"{path}:{line_number}: baris corpus tidak punya field {exc}"
                    raise CorpusError(msg) from exc
        except JsonlDecodeError as exc:
            raise CorpusError(str(exc)) from exc


if __name__ == "__main__":
    documents = list(CorpusLoader().load("tests/fixtures/corpus_sample.jsonl"))
    print(f"{len(documents)} dokumen dimuat")
    for document in documents[:3]:
        print(f"- {document.id} ({document.title}): {document.text[:60]!r}")
