import numpy as np
import pandas as pd
import pytest

from app.inner_speech import _align_common_channels, _load_events


def test_inner_speech_alignment_uses_named_common_channels():
    task = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    baseline = np.arange(5 * 3 * 4, dtype=np.float32).reshape(5, 3, 4)

    task_aligned, baseline_aligned, names = _align_common_channels(
        task,
        ["C3", "C4", "Cz"],
        baseline,
        ["AUX1", "Cz", "C4"],
    )

    assert names == ["C4", "Cz"]
    assert task_aligned.shape == (2, 2, 4)
    assert baseline_aligned.shape == (5, 2, 4)
    np.testing.assert_array_equal(task_aligned[:, 0], task[:, 1])
    np.testing.assert_array_equal(task_aligned[:, 1], task[:, 2])
    np.testing.assert_array_equal(baseline_aligned[:, 0], baseline[:, 2])
    np.testing.assert_array_equal(baseline_aligned[:, 1], baseline[:, 1])


def test_inner_speech_alignment_rejects_channel_set_drift():
    task = np.zeros((1, 2, 4), dtype=np.float32)
    baseline = np.zeros((1, 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="channel set changed"):
        _align_common_channels(
            task,
            ["C3", "C4"],
            baseline,
            ["C3", "C4"],
            expected_order=["C3", "Cz"],
        )


def test_inner_speech_alignment_rejects_duplicate_names():
    task = np.zeros((1, 2, 4), dtype=np.float32)
    baseline = np.zeros((1, 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="duplicate channel names"):
        _align_common_channels(
            task,
            ["C3", "C3"],
            baseline,
            ["C3", "C4"],
        )



def test_current_derivative_event_table_uses_code_and_condition(tmp_path):
    path = tmp_path / "sub-01_ses-01_events.dat"
    frame = pd.DataFrame(
        {
            "Time": [100, 200, 300, 400],
            "Code": [0, 1, 2, 3],
            "condition": [0, 1, 2, 1],
            "block": [1, 1, 1, 1],
        }
    )
    frame.to_csv(path)

    direction, condition = _load_events(path)

    np.testing.assert_array_equal(direction, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(condition, np.array([0, 1, 2, 1]))
