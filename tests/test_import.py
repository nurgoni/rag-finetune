"""Sanity test F0: paket bisa di-import, murah, dan error-nya informatif."""

import logging
import subprocess
import sys

import pytest

import rag_finetune
from rag_finetune.exceptions import MissingDependencyError, RagFinetuneError
from rag_finetune.utils.logging_config import setup_logging


def test_version_is_non_empty_string():
    assert isinstance(rag_finetune.__version__, str)
    assert rag_finetune.__version__


def test_import_is_cheap_and_side_effect_free():
    """Import paket di proses bersih harus sukses tanpa memuat submodul apa pun.

    Ini menjaga exit criteria F0: `import rag_finetune` tidak melakukan I/O.
    Submodul yang bocor ke `sys.modules` adalah tanda ada import di level paket.
    """
    code = (
        "import sys, rag_finetune; "
        "leaked = [m for m in sys.modules if m.startswith('rag_finetune.')]; "
        "print(leaked)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]", f"submodul ter-import saat import paket: {result.stdout}"


def test_missing_dependency_error_names_the_extra():
    """Pesan error harus menyebut paket dan perintah install-nya (ADR-008)."""
    exc = MissingDependencyError(package="faiss", extra="dense")

    assert exc.package == "faiss"
    assert exc.extra == "dense"
    assert "faiss" in str(exc)
    assert "uv sync --extra dense" in str(exc)


def test_all_errors_share_one_base():
    """Boundary sistem harus bisa menangkap semuanya dengan satu except."""
    assert issubclass(MissingDependencyError, RagFinetuneError)


def test_setup_logging_applies_requested_level():
    setup_logging("DEBUG")

    assert logging.getLogger("any.module").getEffectiveLevel() == logging.DEBUG


def test_setup_logging_is_case_insensitive():
    setup_logging("warning")

    assert logging.getLogger("any.module").getEffectiveLevel() == logging.WARNING


def test_setup_logging_rejects_unknown_level():
    """Level salah ketip harus gagal cepat, bukan diam-diam jatuh ke default."""
    with pytest.raises(ValueError, match="TRACE"):
        setup_logging("TRACE")
