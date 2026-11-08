"""Test untuk QAItem (PRD §6.3, FR-2.1)."""

from rag_finetune.schema import QAItem


def test_qa_item_holds_question_and_multiple_golden_answers():
    item = QAItem(
        id="q1", question="Siapa penemu bola lampu?", golden_answers=["Edison", "Thomas Edison"]
    )

    assert item.question == "Siapa penemu bola lampu?"
    assert item.golden_answers == ["Edison", "Thomas Edison"]
