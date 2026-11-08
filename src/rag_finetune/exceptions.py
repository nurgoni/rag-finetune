"""Hierarki exception untuk seluruh repo.

Semua error yang berasal dari kode repo ini turun dari `RagFinetuneError`, supaya
pemanggil di boundary sistem (CLI) bisa `except RagFinetuneError` tanpa ikut menelan
`KeyboardInterrupt` atau bug `AttributeError` kita sendiri.

Modul ini sengaja hanya berisi base class dan satu subclass yang sudah benar-benar
dibutuhkan sekarang. Exception lain (`ConfigError`, `GeneratorError`, ...) lahir
bersama modul yang melemparkannya — mendefinisikannya sekarang berarti menebak
signature konstruktor untuk kode yang belum ada (AGENTS.md §9).
"""


class RagFinetuneError(Exception):
    """Base untuk semua error yang berasal dari repo ini."""


class MissingDependencyError(RagFinetuneError):
    """Dependensi opsional dibutuhkan tapi belum ter-install.

    Dependensi berat (`faiss`, `torch`, `bm25s`) adalah extras opsional dan di-import
    di dalam method, bukan di level modul (ADR-008). Exception ini menyeragamkan
    pesan errornya supaya pengguna langsung tahu extras mana yang harus di-install,
    bukan cuma melihat `ModuleNotFoundError` mentah.
    """

    def __init__(self, package: str, extra: str) -> None:
        """Susun pesan error yang menyebutkan perintah install yang tepat.

        Args:
            package: Nama paket yang gagal di-import, misal `"faiss"`.
            extra: Nama extras yang menyediakannya, misal `"dense"`.
        """
        self.package = package
        self.extra = extra
        super().__init__(f"Paket '{package}' belum ter-install. Jalankan: uv sync --extra {extra}")


if __name__ == "__main__":
    # Demo: bagaimana MissingDependencyError dipakai nanti di retriever/dense.py.
    try:
        raise MissingDependencyError(package="faiss", extra="dense")
    except RagFinetuneError as exc:
        print(f"{type(exc).__name__}: {exc}")
