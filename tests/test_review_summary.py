"""Pure tests for format_review_summary."""

from person_anonymizer.stage_review import format_review_summary

STATS = {"added": 3, "removed": 1, "frames_modified": 1, "frames_reviewed": 7}


def test_format_review_summary_zero_frames_prints_na():
    lines = format_review_summary(STATS, 0, show_reviewed=False)

    assert lines[-1] == "  Frame modificati:      1  (N/A)"


def test_format_review_summary_rounds_percentage_to_one_decimal():
    lines = format_review_summary(STATS, 3, show_reviewed=False)

    assert lines[-1] == "  Frame modificati:      1  (33.3%)"


def test_format_review_summary_web_omits_reviewed_line():
    lines = format_review_summary(STATS, 10, show_reviewed=False)

    assert lines == [
        "\n  Revisione completata:",
        "  Poligoni aggiunti:     3",
        "  Poligoni rimossi:      1",
        "  Frame modificati:      1  (10.0%)",
    ]


def test_format_review_summary_cli_includes_reviewed_line_after_header():
    lines = format_review_summary(STATS, 10, show_reviewed=True)

    assert lines[1] == "  Frame revisionati:     7 / 10"
    assert len(lines) == 5
