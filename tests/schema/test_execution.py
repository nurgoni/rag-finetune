"""Test untuk GenerationStep, JudgeDecision, VoteTally, Aggregation, QAResult."""

import pytest

from rag_finetune.schema import (
    Aggregation,
    GenerationStep,
    JudgeDecision,
    QAItem,
    QAResult,
    VoteTally,
)


def _item() -> QAItem:
    return QAItem(id="q1", question="Siapa penemu bola lampu?", golden_answers=["Thomas Edison"])


def _step(index: int, label: str, query: str, answer: str) -> GenerationStep:
    return GenerationStep(
        index=index,
        label=label,
        query=query,
        prompt=f"Jawab: {query}",
        raw_output=answer,
        answer=answer,
    )


def test_qa_result_derived_properties_point_to_last_step():
    steps = [
        _step(0, "iter-0", "Siapa penemu bola lampu?", "tidak yakin"),
        _step(1, "iter-1", "Siapa penemu bola lampu? tidak yakin", "Thomas Edison"),
    ]
    result = QAResult(item=_item(), steps=steps, prediction="Thomas Edison")

    assert result.last_step is steps[-1]
    assert result.prompt == steps[-1].prompt
    assert result.raw_output == steps[-1].raw_output
    assert result.retrieved == steps[-1].retrieved


def test_qa_result_last_step_raises_on_empty_steps_for_failed_item():
    result = QAResult(item=_item(), steps=[], prediction="", error="generator timeout")

    with pytest.raises(ValueError, match="steps"):
        _ = result.last_step


def test_judge_decision_records_fallback_reason():
    decision = JudgeDecision(
        needs_retrieval=True, reason="output tidak bisa diparse", raw_output="???"
    )

    assert decision.needs_retrieval is True
    assert "tidak bisa diparse" in decision.reason


def test_aggregation_distinguishes_majority_from_tie_break():
    tally = [
        VoteTally(answer="Paris", normalized="paris", votes=1, best_score=0.9),
        VoteTally(answer="London", normalized="london", votes=1, best_score=0.8),
    ]
    aggregation = Aggregation(strategy="majority_vote", tally=tally, resolved_by="tie_break")

    assert aggregation.resolved_by == "tie_break"
    assert len(aggregation.tally) == 2
