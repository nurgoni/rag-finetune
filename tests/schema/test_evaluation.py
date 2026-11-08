"""Test untuk MetricReport (PRD §6.3, FR-8.4)."""

from rag_finetune.schema import MetricReport


def test_metric_report_holds_scores_and_provenance():
    report = MetricReport(
        run_id="abc123",
        scores={"exact_match": 0.4},
        num_items=10,
        avg_llm_calls=1.0,
        provenance={"config_hash": "deadbeef"},
    )

    assert report.scores["exact_match"] == 0.4
    assert report.provenance["config_hash"] == "deadbeef"
