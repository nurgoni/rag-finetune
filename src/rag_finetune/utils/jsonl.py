"""Baca & tulis file JSONL baris demi baris.

`utils/` tidak boleh meng-import modul lain dari repo ini (AGENTS.md §4), jadi
`JsonlDecodeError` di sini berdiri sendiri alih-alih turun dari `RagFinetuneError`.
Pemanggil di `dataset/` atau `corpus/` membungkusnya jadi exception domain masing-masing.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class JsonlDecodeError(Exception):
    """Satu baris JSONL gagal di-parse sebagai JSON."""

    def __init__(self, path: str, line_number: int, original: json.JSONDecodeError) -> None:
        """Susun pesan error yang menyebut file dan nomor baris yang cacat.

        Args:
            path: Path file JSONL yang sedang dibaca.
            line_number: Nomor baris fisik (dimulai dari 1) yang gagal di-parse.
            original: `JSONDecodeError` asli dari modul `json`.
        """
        self.path = path
        self.line_number = line_number
        self.original = original
        super().__init__(f"{path}:{line_number}: baris JSON tidak valid ({original.msg})")


def read_jsonl(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Baca file JSONL baris demi baris, streaming.

    Args:
        path: Path ke file JSONL.

    Yields:
        Tuple `(nomor_baris, baris)` untuk tiap baris yang tidak kosong. Nomor
        baris mengacu ke baris fisik di file (dimulai dari 1), supaya baris
        kosong yang dilewati tidak menggeser penghitungan baris cacat berikutnya.

    Raises:
        JsonlDecodeError: Kalau sebuah baris bukan JSON valid.
    """
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise JsonlDecodeError(path=path, line_number=line_number, original=exc) from exc


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    """Tulis baris-baris dict sebagai JSONL, satu objek JSON per baris.

    Args:
        path: Path file tujuan. Folder induknya dibuat kalau belum ada.
        rows: Baris yang ditulis berurutan, sesuai urutan iterasi `rows`.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        demo_path = f"{tmp_dir}/demo.jsonl"
        write_jsonl(demo_path, [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}])
        for line_number, row in read_jsonl(demo_path):
            print(f"{line_number}: {row}")
