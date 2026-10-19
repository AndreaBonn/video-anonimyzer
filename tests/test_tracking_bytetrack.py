"""
Integration tests for update_tracker against the real ultralytics BYTETracker.

The contract under test: every NMS detection comes back exactly once, in input
order and with its original coordinates (so no person is left unobscured), and
the same person keeps the same track id across frames (so the temporal smoother
never blends two different people).
"""

import logging

import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.tracking import create_tracker, update_tracker

FRAME_SHAPE = (720, 1280, 3)


def _person(x_shift=0, conf=0.9):
    return [100 + x_shift, 100, 200 + x_shift, 300, conf]


def _other_person(conf=0.8):
    return [700, 120, 790, 340, conf]


@pytest.fixture
def tracker():
    return create_tracker(fps=25, config=PipelineConfig())


def test_update_tracker_same_person_keeps_same_id_across_frames(tracker):
    ids = []
    for frame in range(5):
        result = update_tracker(tracker, [_person(x_shift=frame * 3)], FRAME_SHAPE)
        ids.append(result[0][0])

    assert len(set(ids)) == 1


def test_update_tracker_two_people_get_distinct_stable_ids(tracker):
    frames = [
        update_tracker(tracker, [_person(x_shift=f * 2), _other_person()], FRAME_SHAPE)
        for f in range(4)
    ]

    first_ids = [r[0][0] for r in frames]
    second_ids = [r[1][0] for r in frames]
    assert len(set(first_ids)) == 1
    assert len(set(second_ids)) == 1
    assert first_ids[0] != second_ids[0]


def test_update_tracker_id_follows_the_person_not_the_list_position(tracker):
    # Detector output order is arbitrary between frames: an index-based id
    # would hand person A's smoothing state to person B.
    first = update_tracker(tracker, [_person(), _other_person()], FRAME_SHAPE)
    update_tracker(tracker, [_person(x_shift=2), _other_person()], FRAME_SHAPE)

    swapped = update_tracker(tracker, [_other_person(), _person(x_shift=4)], FRAME_SHAPE)

    assert swapped[0][0] == first[1][0]
    assert swapped[1][0] == first[0][0]


def test_update_tracker_returns_every_detection_with_original_coordinates(tracker):
    boxes = [_person(), _other_person()]

    result = update_tracker(tracker, boxes, FRAME_SHAPE)

    assert [(r[1], r[2], r[3], r[4], r[5]) for r in result] == [
        pytest.approx(tuple(b)) for b in boxes
    ]


def test_update_tracker_low_confidence_detection_is_still_returned_with_stable_id(tracker):
    # 0.25 is above the YOLO threshold but below ByteTrack's new-track threshold:
    # ByteTrack alone would drop it, leaving the person unobscured.
    ids = [
        update_tracker(tracker, [_person(x_shift=f * 2, conf=0.25)], FRAME_SHAPE)[0][0]
        for f in range(4)
    ]

    assert len(set(ids)) == 1


def test_update_tracker_new_person_is_returned_on_its_first_frame(tracker):
    update_tracker(tracker, [_person()], FRAME_SHAPE)

    result = update_tracker(tracker, [_person(x_shift=2), _other_person()], FRAME_SHAPE)

    assert len(result) == 2
    assert result[1][1:5] == pytest.approx(tuple(_other_person()[:4]))


def test_update_tracker_confident_person_gets_a_bytetrack_id(tracker):
    # ByteTrack ids are positive, fallback ids negative
    result = update_tracker(tracker, [_person()], FRAME_SHAPE)

    assert result[0][0] > 0


def test_update_tracker_low_confidence_person_gets_a_fallback_id(tracker):
    result = update_tracker(tracker, [_person(conf=0.25)], FRAME_SHAPE)

    assert result[0][0] < 0


def test_update_tracker_new_person_keeps_its_id_when_bytetrack_confirms_it(tracker):
    # From the second frame on, ByteTrack confirms a new person only one frame
    # after it appears: the id must not change at that moment, or the smoother
    # would restart the EMA and leave a ghost of the "old" identity behind.
    update_tracker(tracker, [_person()], FRAME_SHAPE)
    ids = [
        update_tracker(tracker, [_person(x_shift=f * 2), _other_person()], FRAME_SHAPE)[1][0]
        for f in range(1, 5)
    ]

    assert len(set(ids)) == 1


def test_update_tracker_box_far_from_previous_one_gets_a_new_id(tracker):
    first = update_tracker(tracker, [_person(conf=0.25)], FRAME_SHAPE)

    moved = update_tracker(tracker, [_person(x_shift=90, conf=0.25)], FRAME_SHAPE)

    assert moved[0][0] != first[0][0]


def test_update_tracker_failing_tracker_logs_traceback_only_once(tracker, caplog):
    def explode(*_args, **_kwargs):
        raise RuntimeError("tracker crash")

    tracker.byte_tracker.update = explode
    with caplog.at_level(logging.WARNING, logger="person_anonymizer.tracking"):
        for _ in range(3):
            update_tracker(tracker, [_person()], FRAME_SHAPE)

    assert [bool(r.exc_info) for r in caplog.records] == [True, False, False]


def test_update_tracker_logs_no_warning_on_normal_frames(tracker, caplog):
    with caplog.at_level(logging.WARNING, logger="person_anonymizer.tracking"):
        for f in range(3):
            update_tracker(tracker, [_person(x_shift=f)], FRAME_SHAPE)

    assert caplog.records == []


def test_update_tracker_empty_frame_returns_empty_list(tracker):
    assert update_tracker(tracker, [], FRAME_SHAPE) == []


def test_update_tracker_failing_tracker_still_returns_every_box_with_unique_ids(tracker):
    def explode(*_args, **_kwargs):
        raise RuntimeError("tracker crash")

    tracker.byte_tracker.update = explode
    boxes = [_person(), _other_person()]

    result = update_tracker(tracker, boxes, FRAME_SHAPE)

    assert [r[1:] for r in result] == [pytest.approx(tuple(b)) for b in boxes]
    assert result[0][0] != result[1][0]
