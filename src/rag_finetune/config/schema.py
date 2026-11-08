"""Model konfigurasi Pydantic, divalidasi dari YAML (PRD §9, ADR-002).

Setiap sub-model menyimpan hanya field yang dibutuhkan satu komponen, supaya komponen
bisa diberi sub-model-nya saja alih-alih seluruh `Config` (lihat ROADMAP F1 "Catatan desain").
`extra="forbid"` di semua model: field tak dikenal di YAML adalah error, bukan diabaikan
diam-diam (FR-1.1).
"""

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

MAX_SEED = 2**32 - 1


class DatasetConfig(BaseModel):
    """Lokasi dan irisan dataset QA yang dipakai (FR-2.2).

    Args:
        path: Path ke file JSONL dataset.
        split: Irisan dataset yang dipakai.
        limit: Batas jumlah item untuk smoke test. `None` berarti seluruh dataset.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    split: str = "test"
    limit: int | None = None


class CorpusConfig(BaseModel):
    """Lokasi corpus dokumen yang bisa diambil retriever (FR-3.1).

    Args:
        path: Path ke file JSONL corpus.
    """

    model_config = ConfigDict(extra="forbid")

    path: str


class IndexConfig(BaseModel):
    """Lokasi artefak index yang dibangun oleh `rag-finetune index` (FR-3.2).

    Args:
        output_dir: Folder tempat index (dan metadatanya) disimpan.
    """

    model_config = ConfigDict(extra="forbid")

    output_dir: str = "indexes/default"


class RetrieverConfig(BaseModel):
    """Retriever yang dipakai pipeline (FR-4).

    Args:
        type: Nama retriever terdaftar di registry — `"random"`, `"bm25"`, atau `"dense"`.
        top_k: Jumlah dokumen yang diambil per query.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["random", "bm25", "dense"] = "random"
    top_k: int = 5


class GeneratorConfig(BaseModel):
    """Backend generator dan parameter generasinya (FR-6).

    Args:
        type: Nama generator terdaftar di registry — `"echo"` atau `"openai_compatible"`.
        base_url: Base URL endpoint OpenAI-compatible. Dibutuhkan untuk `"openai_compatible"`.
        model: Nama model yang diminta dari generator.
        api_key: Secret dari `.env`, tidak pernah ditulis di YAML yang di-commit (FR-1.3).
        temperature: Parameter sampling generasi.
        max_retries: Batas percobaan ulang saat gagal jaringan (FR-6.4).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["echo", "openai_compatible"] = "echo"
    base_url: str | None = None
    model: str | None = None
    api_key: SecretStr | None = None
    temperature: float = 0.0
    max_retries: int = 3


class PromptConfig(BaseModel):
    """Template prompt yang dipakai pipeline (FR-5).

    Args:
        template: Nama template terdaftar di `prompt/templates/`, misal `"qa"`.
    """

    model_config = ConfigDict(extra="forbid")

    template: str = "qa"


class EvaluationConfig(BaseModel):
    """Metrik yang dihitung saat evaluasi (FR-8.1, FR-8.3).

    Args:
        metrics: Nama metrik terdaftar di registry evaluasi.
    """

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] = Field(default_factory=lambda: ["exact_match", "f1"])


class RunConfig(BaseModel):
    """Parameter yang mengikat satu run secara keseluruhan (FR-1.5, FR-7.2).

    Args:
        seed: Sumber keacakan tunggal untuk seluruh run — `RandomRetriever`, tie-break,
            dan sampling dataset. Tidak ada kode di repo ini yang memanggil `random`
            global (ADR-015).
        pipeline: Arsitektur pipeline yang dijalankan.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, ge=0, le=MAX_SEED)
    pipeline: Literal["sequential", "loop", "branching", "conditional"] = "sequential"


class Config(BaseModel):
    """Konfigurasi penuh satu run, hasil merge `base.yaml` + YAML eksperimen.

    Args:
        dataset: Lihat `DatasetConfig`.
        corpus: Lihat `CorpusConfig`.
        index: Lihat `IndexConfig`.
        retriever: Lihat `RetrieverConfig`.
        generator: Lihat `GeneratorConfig`.
        prompt: Lihat `PromptConfig`.
        evaluation: Lihat `EvaluationConfig`.
        run: Lihat `RunConfig`.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    corpus: CorpusConfig
    index: IndexConfig = Field(default_factory=IndexConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    def config_hash(self) -> str:
        """Hash deterministik atas seluruh field config yang sudah di-resolve.

        Dua config yang berbeda di field apa pun — termasuk default yang ikut
        ter-resolve — menghasilkan hash berbeda; dua config identik menghasilkan
        hash sama (FR-1.6, ADR-016). `SecretStr` ikut di-mask secara konsisten oleh
        `model_dump_json`, jadi nilai secret tidak memengaruhi hash — itu memang
        yang diinginkan, karena secret bukan bagian provenance eksperimen.

        Returns:
            Digest SHA-256 heksadesimal atas representasi JSON config dengan
            key terurut.
        """
        canonical_json = self.model_dump_json()
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    config = Config(
        dataset=DatasetConfig(path="datasets/nq/test.jsonl"),
        corpus=CorpusConfig(path="datasets/nq/corpus.jsonl"),
    )
    print(config.model_dump_json(indent=2))
    print(f"config_hash={config.config_hash()}")
