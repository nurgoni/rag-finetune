"""Model data untuk satu item dataset QA (PRD §6.3, FR-2.1)."""

from pydantic import BaseModel


class QAItem(BaseModel, frozen=True):
    """Satu pertanyaan dari dataset QA beserta jawaban acuannya.

    Args:
        id: Pengenal unik item di dalam dataset.
        question: Pertanyaan dalam natural language.
        golden_answers: Jawaban acuan yang dianggap benar. Bisa lebih dari
            satu varian permukaan (misal "JFK" dan "John F. Kennedy").
    """

    id: str
    question: str
    golden_answers: list[str]


if __name__ == "__main__":
    item = QAItem(id="q1", question="Siapa penemu bola lampu?", golden_answers=["Thomas Edison"])
    print(item.model_dump_json(indent=2))
