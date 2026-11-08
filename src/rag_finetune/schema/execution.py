"""Model data untuk jejak eksekusi satu query lewat pipeline (PRD §6.3, §6.3.1).

`QAResult.steps` adalah daftar sejak F3 (ADR-011) karena loop dan branching memecahkan
asumsi "satu query = satu kali retrieve + satu kali generate". `last_step`, `retrieved`,
`prompt`, dan `raw_output` di `QAResult` adalah satu-satunya logika yang diizinkan hidup
di dalam model `schema/` — keduanya ada supaya kode `SequentialPipeline` (yang selalu
hanya mengisi satu langkah) tidak perlu menulis `steps[-1]` berulang-ulang.
"""

from typing import Literal

from pydantic import BaseModel, Field

from rag_finetune.schema.document import RetrievedDocument
from rag_finetune.schema.qa import QAItem


class GenerationStep(BaseModel, frozen=True):
    """Satu putaran retrieve (opsional) + generate di dalam sebuah pipeline.

    Args:
        index: Posisi langkah ini di dalam `QAResult.steps`, dimulai dari 0.
        label: Nama yang menerangkan peran langkah ini, misal `"iter-0"` (loop)
            atau `"doc-2"` (branching). Sequential selalu memakai `"main"`.
        query: Query yang benar-benar dipakai langkah ini. Berbeda dari
            `QAItem.question` di loop, yang menyusun ulang query tiap iterasi.
        retrieved: Dokumen hasil retrieval untuk langkah ini. Kosong pada
            cabang no-retrieval (conditional).
        prompt: Prompt final yang dikirim ke generator.
        raw_output: Teks mentah yang dikembalikan generator, sebelum di-parse.
        answer: Jawaban yang sudah diekstrak dari `raw_output`.
    """

    index: int
    label: str
    query: str
    retrieved: list[RetrievedDocument] = Field(default_factory=list)
    prompt: str
    raw_output: str
    answer: str


class JudgeDecision(BaseModel, frozen=True):
    """Keputusan `BaseJudger` tentang apakah sebuah query butuh retrieval.

    Args:
        needs_retrieval: `True` kalau judger memutuskan query ini perlu retrieval.
        reason: Alasan singkat, termasuk kasus fallback saat output judger
            tak bisa di-parse (FR-7d.3).
        raw_output: Output mentah judger sebelum di-parse. `None` untuk judger
            yang tidak berbasis LLM (`AlwaysRetrieveJudger`, `NeverRetrieveJudger`).
    """

    needs_retrieval: bool
    reason: str
    raw_output: str | None = None


class VoteTally(BaseModel, frozen=True):
    """Hitungan vote untuk satu kandidat jawaban pada pipeline branching.

    Args:
        answer: Bentuk asli kandidat jawaban, sebelum normalisasi.
        normalized: Bentuk ternormalisasi yang dipakai untuk mencocokkan vote
            (lihat `evaluation.normalize_answer`).
        votes: Jumlah kandidat yang jatuh ke bentuk ternormalisasi ini.
        best_score: Skor retrieval tertinggi di antara dokumen pendukung
            kandidat ini. Dipakai untuk memutus seri (FR-7c.4).
    """

    answer: str
    normalized: str
    votes: int
    best_score: float


class Aggregation(BaseModel, frozen=True):
    """Hasil agregasi beberapa kandidat jawaban jadi satu prediksi (branching).

    Args:
        strategy: Nama strategi agregasi yang dipakai, misal `"majority_vote"`.
        tally: Tally lengkap tiap kandidat jawaban (FR-7c.7).
        resolved_by: `"majority"` kalau ada pemenang jelas, `"tie_break"` kalau
            diputus lewat skor retrieval tertinggi karena seri.
    """

    strategy: str
    tally: list[VoteTally]
    resolved_by: Literal["majority", "tie_break"]


class QAResult(BaseModel, frozen=True):
    """Hasil lengkap satu item QA setelah melewati sebuah pipeline.

    Args:
        item: Item QA asal.
        steps: Jejak tiap putaran retrieve+generate. Tidak boleh kosong kecuali
            `error` terisi (item gagal sebelum langkah pertama selesai).
        prediction: Jawaban akhir setelah seluruh langkah/agregasi.
        judge: Keputusan judger. Hanya terisi pada pipeline conditional.
        aggregation: Hasil agregasi. Hanya terisi pada pipeline branching.
        num_llm_calls: Jumlah panggilan generator yang benar-benar terjadi,
            dihitung di boundary generator (ADR-014), bukan oleh pipeline.
        elapsed_s: Durasi eksekusi item ini dalam detik.
        error: Pesan error kalau item ini gagal (FR-7.4). `None` kalau sukses.
    """

    item: QAItem
    steps: list[GenerationStep] = Field(default_factory=list)
    prediction: str
    judge: JudgeDecision | None = None
    aggregation: Aggregation | None = None
    num_llm_calls: int = 0
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def last_step(self) -> GenerationStep:
        """Langkah terakhir yang tercatat.

        Returns:
            Elemen terakhir dari `steps`.

        Raises:
            ValueError: Kalau `steps` kosong — biasanya berarti item ini gagal
                sebelum langkah pertama selesai; periksa `error` sebagai gantinya.
        """
        if not self.steps:
            msg = "QAResult tidak punya steps; periksa `error` untuk item yang gagal"
            raise ValueError(msg)
        return self.steps[-1]

    @property
    def retrieved(self) -> list[RetrievedDocument]:
        """Dokumen terambil pada langkah terakhir."""
        return self.last_step.retrieved

    @property
    def prompt(self) -> str:
        """Prompt final pada langkah terakhir."""
        return self.last_step.prompt

    @property
    def raw_output(self) -> str:
        """Output mentah generator pada langkah terakhir."""
        return self.last_step.raw_output


if __name__ == "__main__":
    item = QAItem(id="q1", question="Siapa penemu bola lampu?", golden_answers=["Thomas Edison"])
    step = GenerationStep(
        index=0,
        label="main",
        query=item.question,
        prompt=f"Jawab: {item.question}",
        raw_output="Thomas Edison",
        answer="Thomas Edison",
    )
    result = QAResult(item=item, steps=[step], prediction=step.answer, num_llm_calls=1)
    print(f"prediction={result.prediction!r} prompt={result.prompt!r}")
