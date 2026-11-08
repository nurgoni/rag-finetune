"""Model data lintas-layer. Hanya Pydantic + stdlib (AGENTS.md §4)."""

from rag_finetune.schema.document import Document, RetrievedDocument
from rag_finetune.schema.evaluation import MetricReport
from rag_finetune.schema.execution import (
    Aggregation,
    GenerationStep,
    JudgeDecision,
    QAResult,
    VoteTally,
)
from rag_finetune.schema.qa import QAItem

__all__ = [
    "Aggregation",
    "Document",
    "GenerationStep",
    "JudgeDecision",
    "MetricReport",
    "QAItem",
    "QAResult",
    "RetrievedDocument",
    "VoteTally",
]
