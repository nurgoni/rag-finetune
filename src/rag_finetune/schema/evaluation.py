"""Model data untuk hasil agregat evaluasi satu run (PRD §6.3, FR-8.4)."""

from pydantic import BaseModel, Field


class MetricReport(BaseModel, frozen=True):
    """Ringkasan metrik satu run, ditulis ke `results/<run_id>/metrics.json`.

    Args:
        run_id: Pengenal run, deterministik dari hash config + timestamp.
        scores: Skor kualitas per metrik, misal `{"exact_match": 0.42, "f1": 0.51}`.
        num_items: Jumlah item yang dievaluasi.
        avg_llm_calls: Rata-rata `QAResult.num_llm_calls` di seluruh item (FR-8.5).
        provenance: Informasi yang cukup untuk membedakan dua run mana pun —
            hash config, parameter pipeline, tipe retriever, nama model,
            `temperature`, `seed`, dan hash template prompt (ADR-016).
    """

    run_id: str
    scores: dict[str, float] = Field(default_factory=dict)
    num_items: int
    avg_llm_calls: float
    provenance: dict[str, str] = Field(default_factory=dict)


if __name__ == "__main__":
    report = MetricReport(
        run_id="abc123",
        scores={"exact_match": 0.42, "f1": 0.51},
        num_items=50,
        avg_llm_calls=1.0,
        provenance={"config_hash": "deadbeef", "pipeline.type": "sequential"},
    )
    print(report.model_dump_json(indent=2))
