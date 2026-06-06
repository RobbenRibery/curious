from curious.utils.utils import (
    build_completion_sample_rows,
    format_completion_sample_rows,
    log_completion_sample_table,
)


def test_build_completion_sample_rows_is_bounded_and_offsets_indices():
    rows = build_completion_sample_rows(
        phase="eval",
        batch_idx=10,
        questions=["q1", "q2", "q3"],
        answers=["a1", "a2", "a3"],
        completions=["c1", "c2", "c3"],
        rewards=[1.0, 0.0, -1.0],
        infos=[{"ok": True}, {"ok": False}, {"ok": False}],
        max_samples=2,
        sample_offset=5,
    )

    assert [row["sample_idx"] for row in rows] == [5, 6]
    assert [row["completion"] for row in rows] == ["c1", "c2"]
    assert rows[0]["phase"] == "eval"
    assert rows[0]["batch_idx"] == 10


def test_format_completion_sample_rows_uses_logging_template():
    rows = build_completion_sample_rows(
        phase="train",
        batch_idx=3,
        questions=["What is 1+1?"],
        answers=["2"],
        completions=["The answer is 2."],
        rewards=[1.0],
        infos=[{"outcome_reward": 1.0}],
        max_samples=1,
    )

    text = format_completion_sample_rows(rows)

    assert "Question:" in text
    assert "What is 1+1?" in text
    assert "The answer is 2." in text


def test_log_completion_sample_table_keeps_training_step():
    captured = []
    rows = build_completion_sample_rows(
        phase="train",
        batch_idx=7,
        questions=["q"],
        answers=["a"],
        completions=["c"],
        rewards=[0.5],
        infos=[{}],
        max_samples=1,
    )

    log_completion_sample_table(
        logger=captured.append,
        key="train/completion_samples",
        rows=rows,
    )

    assert captured[0]["num_batches_visited"] == 7
    assert "train/completion_samples" in captured[0]
