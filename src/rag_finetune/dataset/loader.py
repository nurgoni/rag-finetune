"""Baca dataset QA dari JSONL jadi `QAItem` (FR-2)."""

from pydantic import ValidationError

from rag_finetune.exceptions import RagFinetuneError
from rag_finetune.schema import QAItem
from rag_finetune.utils.jsonl import JsonlDecodeError, read_jsonl


class DatasetError(RagFinetuneError):
    """Dataset QA gagal dibaca atau divalidasi."""


class QADatasetLoader:
    """Memuat dataset QA dari satu file JSONL jadi `list[QAItem]`.

    Satu file JSONL diasumsikan sudah berisi satu split (FR-2.2) — konvensi
    penamaan file (misal `train.jsonl`, `test.jsonl`) menentukan split-nya,
    bukan sebuah field di dalam baris. `DatasetConfig.split` dipakai untuk
    provenance, bukan untuk memfilter baris di dalam loader ini.
    """

    def load(self, path: str, limit: int | None = None) -> list[QAItem]:
        """Baca JSONL dan validasi tiap baris jadi `QAItem`.

        Args:
            path: Path ke file JSONL dataset.
            limit: Batas jumlah item yang dimuat. `None` berarti seluruhnya.

        Returns:
            Daftar `QAItem`, urut sesuai urutan baris di file.

        Raises:
            DatasetError: Kalau sebuah baris bukan JSON valid, atau tidak
                punya field `id`/`question`/`golden_answers` yang dibutuhkan
                `QAItem`. Pesan menyebut path file dan nomor baris (FR-2.3).
        """
        items: list[QAItem] = []
        try:
            for line_number, row in read_jsonl(path):
                try:
                    items.append(QAItem.model_validate(row))
                except ValidationError as exc:
                    msg = f"{path}:{line_number}: baris tidak valid sebagai QAItem ({exc})"
                    raise DatasetError(msg) from exc
                if limit is not None and len(items) >= limit:
                    break
        except JsonlDecodeError as exc:
            raise DatasetError(str(exc)) from exc
        return items


if __name__ == "__main__":
    loaded_items = QADatasetLoader().load("tests/fixtures/qa_sample.jsonl")
    print(f"{len(loaded_items)} item dimuat")
    for loaded_item in loaded_items[:3]:
        print(f"- {loaded_item.id}: {loaded_item.question!r} -> {loaded_item.golden_answers}")
