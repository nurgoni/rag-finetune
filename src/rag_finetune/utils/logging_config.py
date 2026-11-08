"""Konfigurasi logging terpusat.

Satu-satunya tempat di repo ini yang memanggil `logging.basicConfig`. Modul lain
cukup `logger = logging.getLogger(__name__)` dan tidak mengonfigurasi apa pun.

Namanya `logging_config`, bukan `logging`, supaya tidak menyadur nama modul stdlib.
Modul bernama `logging.py` membuat `import logging` di dalamnya resolve ke dirinya
sendiri begitu file-nya dimuat sebagai modul top-level — dan kegagalannya tersembunyi,
karena di dalam paket absolute import tetap menemukan stdlib dengan benar.
"""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"
DEFAULT_LEVEL = "INFO"
VALID_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def setup_logging(level: str = DEFAULT_LEVEL) -> None:
    """Pasang handler logging untuk seluruh proses.

    Dipilih sebagai function, bukan class, karena ia tidak menyimpan state dan tidak
    punya dependency — hanya mengonfigurasi modul `logging` stdlib sekali jalan
    (AGENTS.md §3).

    Args:
        level: Nama level logging, case-insensitive. Salah satu dari `VALID_LEVELS`.

    Raises:
        ValueError: Kalau `level` bukan nama level yang dikenal. Gagal cepat di sini
            lebih baik daripada diam-diam jatuh ke level default, karena run yang
            seharusnya verbose jadi senyap tanpa penjelasan.

    Examples:
        >>> setup_logging("DEBUG")
        >>> logging.getLogger(__name__).getEffectiveLevel() == logging.DEBUG
        True
    """
    normalized = level.upper()
    if normalized not in VALID_LEVELS:
        valid = ", ".join(sorted(VALID_LEVELS))
        raise ValueError(f"Level logging '{level}' tidak dikenal. Pilihan: {valid}")

    # force=True supaya pemanggilan berulang (CLI lalu test) benar-benar mengganti
    # konfigurasi, bukan diabaikan karena handler sudah ada.
    logging.basicConfig(
        level=normalized,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )


if __name__ == "__main__":
    setup_logging("DEBUG")
    logger = logging.getLogger("demo")
    logger.debug("pesan debug — kelihatan karena level DEBUG")
    logger.info("memuat %d dokumen", 42)
    logger.warning("index belum dibangun")

    setup_logging("WARNING")
    logger.info("pesan ini TIDAK kelihatan — level sudah WARNING")
    logger.warning("hanya warning ke atas yang lolos")
