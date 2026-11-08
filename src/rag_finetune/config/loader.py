"""Baca YAML, gabungkan `base.yaml` + eksperimen, dan validasi jadi `Config` (FR-1.1, FR-1.2)."""

import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from rag_finetune.config.schema import Config
from rag_finetune.exceptions import RagFinetuneError


class ConfigError(RagFinetuneError):
    """Config YAML gagal divalidasi terhadap `Config`."""

    def __init__(self, source: str, original: ValidationError) -> None:
        """Susun pesan error yang menyebut path field yang salah (FR-1.4).

        Args:
            source: Path file YAML yang gagal divalidasi.
            original: `ValidationError` asli dari Pydantic.
        """
        self.source = source
        self.original = original
        field_errors = "\n".join(
            f"  - {'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in original.errors()
        )
        message = f"Config tidak valid ({source}):\n{field_errors}"
        super().__init__(message)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Gabungkan dua dict secara rekursif; `override` menang di key yang sama.

    Args:
        base: Dict dasar, misal isi `base.yaml`.
        override: Dict yang menimpa sebagian isi `base`, misal YAML eksperimen.

    Returns:
        Dict baru hasil penggabungan. Key yang ada di keduanya dan bernilai
        dict digabung rekursif; selain itu, nilai dari `override` menang.
        List tidak digabung — `override` menggantikan seluruh list di `base`.
    """
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


class ConfigLoader:
    """Memuat `Config` tervalidasi dari `base.yaml` dan (opsional) YAML eksperimen."""

    def load(self, base_path: str, experiment_path: str | None = None) -> Config:
        """Baca, gabungkan dalam-dalam, lalu validasi jadi `Config`.

        Args:
            base_path: Path ke YAML dasar (biasanya `configs/base.yaml`).
            experiment_path: Path ke YAML eksperimen yang menuliskan selisihnya
                dari dasar. `None` berarti hanya dasar yang dipakai.

        Returns:
            `Config` yang sudah tervalidasi.

        Raises:
            ConfigError: Kalau hasil gabungan gagal validasi Pydantic — field
                tak dikenal, tipe salah, atau field wajib tak terisi.
        """
        merged = self._read_yaml(base_path)
        if experiment_path is not None:
            merged = deep_merge(merged, self._read_yaml(experiment_path))

        source = experiment_path or base_path
        try:
            return Config.model_validate(merged)
        except ValidationError as exc:
            raise ConfigError(source=source, original=exc) from exc

    @staticmethod
    def _read_yaml(path: str) -> dict[str, Any]:
        """Baca satu file YAML jadi dict.

        Args:
            path: Path ke file YAML.

        Returns:
            Isi file sebagai dict. Dict kosong kalau file YAML kosong.
        """
        content = Path(path).read_text(encoding="utf-8")
        return yaml.safe_load(content) or {}


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/nq_sequential.yaml"
    loaded_config = ConfigLoader().load("configs/base.yaml", config_path)
    print(loaded_config.model_dump_json(indent=2))
    print(f"config_hash={loaded_config.config_hash()}")
