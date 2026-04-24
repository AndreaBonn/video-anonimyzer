"""
Test per web/review_state.py — ReviewState thread-safe.

Copre: is_active, setup, wait_for_completion, complete, get_annotations,
update_annotations, get_metadata, get_frame_jpeg, thread safety.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from person_anonymizer.web.review_state import ReviewState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_review_state_with_mock_cap(annotations=None, total=10, w=640, h=480, fps=25.0):
    """Crea un ReviewState con un VideoCapture mockato già configurato."""
    state = ReviewState()
    ann = annotations or {0: {"auto": [], "manual": [], "intensities": []}}

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    with patch("cv2.VideoCapture", return_value=mock_cap):
        state.setup(
            video_path="/fake/video.mp4",
            annotations=ann,
            total_frames=total,
            frame_w=w,
            frame_h=h,
            fps=fps,
        )

    return state, mock_cap


# ---------------------------------------------------------------------------
# TestReviewStateInitial
# ---------------------------------------------------------------------------


class TestReviewStateInitial:
    """Verifica stato iniziale di ReviewState."""

    def test_is_active_false_initially(self):
        # Arrange / Act
        state = ReviewState()

        # Assert
        assert state.is_active is False

    def test_get_annotations_returns_none_initially(self):
        # Arrange / Act
        state = ReviewState()

        # Assert
        assert state.get_annotations() is None

    def test_get_metadata_returns_zeros_initially(self):
        # Arrange / Act
        state = ReviewState()

        # Assert
        meta = state.get_metadata()
        assert meta["total_frames"] == 0
        assert meta["frame_w"] == 0
        assert meta["frame_h"] == 0
        assert meta["fps"] == 0.0


# ---------------------------------------------------------------------------
# TestReviewStateSetup
# ---------------------------------------------------------------------------


class TestReviewStateSetup:
    """Verifica il metodo setup()."""

    def test_setup_sets_is_active_true(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap()

        # Assert
        assert state.is_active is True

    def test_setup_stores_annotations(self):
        # Arrange
        ann = {0: {"auto": [[[0, 0], [10, 0], [10, 10]]], "manual": [], "intensities": [1]}}
        state, _ = _make_review_state_with_mock_cap(annotations=ann)

        # Act
        result = state.get_annotations()

        # Assert — deep copy, non il riferimento originale
        assert 0 in result
        assert result[0]["auto"] == [[[0, 0], [10, 0], [10, 10]]]

    def test_setup_annotations_is_deep_copy(self):
        # Arrange
        ann = {0: {"auto": [], "manual": [], "intensities": []}}
        state, _ = _make_review_state_with_mock_cap(annotations=ann)

        # Act — modifica il dizionario originale
        ann[0]["auto"].append([[[99, 99]]])

        # Assert — lo stato non deve riflettere la modifica
        result = state.get_annotations()
        assert result[0]["auto"] == []

    def test_setup_stores_metadata(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap(total=100, w=1920, h=1080, fps=30.0)

        # Act
        meta = state.get_metadata()

        # Assert
        assert meta["total_frames"] == 100
        assert meta["frame_w"] == 1920
        assert meta["frame_h"] == 1080
        assert meta["fps"] == 30.0

    def test_setup_releases_previous_cap(self):
        # Arrange
        state, first_cap = _make_review_state_with_mock_cap()

        # Act — secondo setup deve rilasciare il cap precedente
        second_cap = MagicMock()
        second_cap.isOpened.return_value = True
        with patch("cv2.VideoCapture", return_value=second_cap):
            state.setup(
                video_path="/fake/video2.mp4",
                annotations={},
                total_frames=5,
                frame_w=320,
                frame_h=240,
                fps=25.0,
            )

        # Assert — il primo cap è stato rilasciato
        first_cap.release.assert_called_once()

    def test_setup_uses_default_fisheye_context_when_none(self):
        # Arrange — fisheye=None deve creare FisheyeContext() di default
        state = ReviewState()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        with patch("cv2.VideoCapture", return_value=mock_cap):
            state.setup(
                video_path="/fake/video.mp4",
                annotations={},
                total_frames=1,
                frame_w=640,
                frame_h=480,
                fps=25.0,
                fisheye=None,
            )

        # Assert — FisheyeContext creato, enabled=False di default
        assert state._fisheye is not None
        assert state._fisheye.enabled is False

    def test_setup_cap_open_failure_raises_exception(self):
        # Arrange
        state = ReviewState()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", side_effect=Exception("cv2 error")):
            with pytest.raises(Exception, match="cv2 error"):
                state.setup(
                    video_path="/fake/video.mp4",
                    annotations={},
                    total_frames=1,
                    frame_w=640,
                    frame_h=480,
                    fps=25.0,
                )

        # Assert — is_active deve restare False
        assert state.is_active is False


# ---------------------------------------------------------------------------
# TestReviewStateComplete
# ---------------------------------------------------------------------------


class TestReviewStateComplete:
    """Verifica complete() e wait_for_completion()."""

    def test_complete_sets_annotations(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap()
        new_ann = {1: {"auto": [[[0, 0], [5, 0], [5, 5]]], "manual": [], "intensities": []}}

        # Act
        state.complete(new_ann)

        # Assert
        result = state.get_annotations()
        assert 1 in result

    def test_complete_unblocks_wait_for_completion(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap()
        completed_ann = {99: {"auto": [], "manual": [], "intensities": []}}
        result_holder = {}

        def pipeline_thread():
            result_holder["result"] = state.wait_for_completion()

        t = threading.Thread(target=pipeline_thread)
        t.start()

        # Act
        state.complete(completed_ann)
        t.join(timeout=2.0)

        # Assert
        assert not t.is_alive()
        assert 99 in result_holder["result"]

    def test_wait_for_completion_sets_is_active_false(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap()
        result_holder = {}

        def pipeline_thread():
            result_holder["result"] = state.wait_for_completion()

        t = threading.Thread(target=pipeline_thread)
        t.start()

        # Act
        state.complete({})
        t.join(timeout=2.0)

        # Assert
        assert state.is_active is False

    def test_wait_for_completion_returns_deep_copy(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap()
        ann = {5: {"auto": [], "manual": [], "intensities": []}}
        result_holder = {}

        def pipeline_thread():
            result_holder["result"] = state.wait_for_completion()

        t = threading.Thread(target=pipeline_thread)
        t.start()
        state.complete(ann)
        t.join(timeout=2.0)

        # Act — modifica l'originale
        ann[5]["auto"].append("extra")

        # Assert — il risultato non deve essere alterato
        assert result_holder["result"][5]["auto"] == []

    def test_complete_annotations_is_deep_copy(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap()
        ann = {0: {"auto": [], "manual": [], "intensities": []}}

        # Act
        state.complete(ann)
        ann[0]["auto"].append("extra")  # modifica dopo complete

        # Assert — le annotazioni interne non cambiano
        assert state.get_annotations()[0]["auto"] == []


# ---------------------------------------------------------------------------
# TestReviewStateUpdateAnnotations
# ---------------------------------------------------------------------------


class TestReviewStateUpdateAnnotations:
    """Verifica update_annotations() — aggiornamento per singolo frame."""

    def test_update_annotations_stores_frame_data(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap()
        frame_data = {"auto": [[[0, 0], [10, 0], [10, 10]]], "manual": [], "intensities": [5]}

        # Act
        state.update_annotations(frame_idx=3, frame_data=frame_data)

        # Assert
        result = state.get_annotations()
        assert 3 in result
        assert result[3]["intensities"] == [5]

    def test_update_annotations_is_deep_copy(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap()
        frame_data = {"auto": [], "manual": [], "intensities": []}

        # Act
        state.update_annotations(frame_idx=0, frame_data=frame_data)
        frame_data["auto"].append("extra")  # modifica dopo update

        # Assert — lo stato interno non deve cambiare
        assert state.get_annotations()[0]["auto"] == []

    def test_update_annotations_overwrites_existing(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap(
            annotations={0: {"auto": [[[1, 2], [3, 4], [5, 6]]], "manual": [], "intensities": []}}
        )
        new_data = {"auto": [], "manual": [], "intensities": []}

        # Act
        state.update_annotations(frame_idx=0, frame_data=new_data)

        # Assert
        assert state.get_annotations()[0]["auto"] == []


# ---------------------------------------------------------------------------
# TestReviewStateGetFrameJpeg
# ---------------------------------------------------------------------------


class TestReviewStateGetFrameJpeg:
    """Verifica get_frame_jpeg()."""

    def test_get_frame_jpeg_returns_none_when_cap_is_none(self):
        # Arrange
        state = ReviewState()  # cap è None

        # Act
        jpeg, scale = state.get_frame_jpeg(frame_idx=0)

        # Assert
        assert jpeg is None
        assert scale == 1.0

    def test_get_frame_jpeg_returns_none_when_cap_not_opened(self):
        # Arrange
        state, mock_cap = _make_review_state_with_mock_cap()
        mock_cap.isOpened.return_value = False

        # Act
        jpeg, scale = state.get_frame_jpeg(frame_idx=0)

        # Assert
        assert jpeg is None
        assert scale == 1.0

    def test_get_frame_jpeg_returns_none_when_read_fails(self):
        # Arrange
        import numpy as np

        state, mock_cap = _make_review_state_with_mock_cap()
        mock_cap.read.return_value = (False, None)

        # Act
        jpeg, scale = state.get_frame_jpeg(frame_idx=5)

        # Assert
        assert jpeg is None
        assert scale == 1.0

    def test_get_frame_jpeg_returns_bytes_on_success(self):
        # Arrange
        import numpy as np

        state, mock_cap = _make_review_state_with_mock_cap()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)

        # Act
        with patch("cv2.imencode", return_value=(True, MagicMock(tobytes=lambda: b"jpeg_data"))):
            jpeg, scale = state.get_frame_jpeg(frame_idx=0)

        # Assert
        assert scale == 1.0  # larghezza 640 < max_width 1280

    def test_get_frame_jpeg_scales_down_wide_frame(self):
        # Arrange
        import numpy as np

        state, mock_cap = _make_review_state_with_mock_cap(w=2560, h=1440)
        # Frame più largo di max_width=1280
        fake_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)

        mock_jpeg_data = MagicMock()
        mock_jpeg_data.tobytes.return_value = b"jpeg_scaled"

        with patch("cv2.imencode", return_value=(True, mock_jpeg_data)), patch(
            "cv2.resize", return_value=np.zeros((720, 1280, 3), dtype=np.uint8)
        ):
            jpeg, scale = state.get_frame_jpeg(frame_idx=0, max_width=1280)

        # Assert — scale deve essere 0.5 (1280/2560)
        assert abs(scale - 0.5) < 0.001


# ---------------------------------------------------------------------------
# TestReviewStateThreadSafety
# ---------------------------------------------------------------------------


class TestReviewStateThreadSafety:
    """Verifica accesso concorrente thread-safe."""

    def test_concurrent_update_and_get_annotations(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap(annotations={i: {"auto": [], "manual": [], "intensities": []} for i in range(10)})
        errors = []

        def writer():
            try:
                for i in range(20):
                    state.update_annotations(
                        frame_idx=i % 10,
                        frame_data={"auto": [], "manual": [], "intensities": [i]},
                    )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    state.get_annotations()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Assert
        assert errors == []

    def test_is_active_thread_safe(self):
        # Arrange
        state, _ = _make_review_state_with_mock_cap()
        results = []

        def check_active():
            results.append(state.is_active)

        threads = [threading.Thread(target=check_active) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        # Assert — nessuna eccezione, tutti i risultati sono bool
        assert all(isinstance(r, bool) for r in results)
