"""Characterization tests for the SAM3 backend, run against a fake ``sam3`` package.

The real ``sam3`` library is not installed: a fake module is injected through
``sys.modules`` and exposes the names the backend imports (``build_sam3``,
``SAM3ImagePredictor``, ``SAM3VideoPredictor``). Masks are real ``torch`` tensors
because the backend calls ``.cpu().numpy()`` on them.
"""

import sys
import threading
import types

import numpy as np
import pytest
import torch

from person_anonymizer import sam3_backend
from person_anonymizer.config import PipelineConfig
from person_anonymizer.sam3_backend import Sam3ImageRefiner, Sam3VideoDetector

H, W = 200, 300

# ─── Fakes ─────────────────────────────────────────────────────


class FakeModel:
    def __init__(self, path):
        self.path = path
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class FakeImagePredictor:
    """Returns a preset mask per box ``(x1, y1, x2, y2)``; unknown boxes get an empty mask."""

    def __init__(self, model):
        self.model = model
        self.images = []
        self.predict_calls = []
        self.masks_by_box = {}

    def set_image(self, image):
        self.images.append(image)

    def predict(self, box, multimask_output):
        self.predict_calls.append((box, multimask_output))
        key = tuple(int(v) for v in box[0].tolist())
        mask = self.masks_by_box.get(key, np.zeros((H, W), dtype=bool))
        masks = torch.from_numpy(mask)[None]  # (1, H, W): one mask since multimask_output=False
        return masks, torch.ones(1), torch.zeros(1)


class FakeVideoPredictor:
    def __init__(self, model):
        self.model = model
        self.frames = []  # list of (frame_idx, bool ndarray (N, H, W))
        self.stop_after_first = None
        self.init_path = None
        self.prompt = None
        self.reset_with = None
        self.yielded = 0

    def init_state(self, video_path):
        self.init_path = video_path
        return {"state_for": video_path}

    def add_new_text_prompt(self, state, text):
        self.prompt = text

    def propagate_in_video(self, state):
        for position, (idx, masks) in enumerate(self.frames):
            if position == 1 and self.stop_after_first is not None:
                self.stop_after_first.set()
            self.yielded += 1
            tensor = torch.from_numpy(masks)[:, None]  # (N, 1, H, W)
            yield idx, list(range(len(masks))), tensor

    def reset_state(self, state):
        self.reset_with = state


@pytest.fixture
def fake_sam3(monkeypatch):
    """Injects a fake ``sam3`` module and returns it; created instances are recorded."""
    module = types.ModuleType("sam3")
    module.models = []
    module.image_predictors = []
    module.video_predictors = []

    def build_sam3(path):
        model = FakeModel(path)
        module.models.append(model)
        return model

    def make_image_predictor(model):
        predictor = FakeImagePredictor(model)
        predictor.masks_by_box = module.masks_by_box
        module.image_predictors.append(predictor)
        return predictor

    def make_video_predictor(model):
        predictor = FakeVideoPredictor(model)
        predictor.frames = module.video_frames
        module.video_predictors.append(predictor)
        return predictor

    module.masks_by_box = {}
    module.video_frames = []
    module.build_sam3 = build_sam3
    module.SAM3ImagePredictor = make_image_predictor
    module.SAM3VideoPredictor = make_video_predictor

    monkeypatch.setitem(sys.modules, "sam3", module)
    monkeypatch.setattr(sam3_backend, "_require_sam3", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return module


def rect_mask(rows, cols):
    mask = np.zeros((H, W), dtype=bool)
    mask[rows[0] : rows[1], cols[0] : cols[1]] = True
    return mask


def rect_corners(x1, y1, x2, y2):
    """Corner set of a filled rectangle covering [x1, x2] x [y1, y2] inclusive."""
    return {(x1, y1), (x2, y1), (x2, y2), (x1, y2)}


# ─── Sam3ImageRefiner.refine_boxes ─────────────────────────────


class TestRefineBoxes:
    def test_mask_inside_box_becomes_polygon_of_mask_not_box(self, fake_sam3):
        # Arrange: mask is narrower than the box
        fake_sam3.masks_by_box[(20, 10, 120, 150)] = rect_mask((20, 120), (30, 70))
        refiner = Sam3ImageRefiner(device="cpu")
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Act
        polygons = refiner.refine_boxes(frame, [(20, 10, 120, 150)])

        # Assert: corners of the mask (cols 30..69, rows 20..119), not of the box
        assert len(polygons) == 1
        assert set(polygons[0]) == rect_corners(30, 20, 69, 119)

    def test_empty_mask_falls_back_to_box_rectangle(self, fake_sam3):
        refiner = Sam3ImageRefiner(device="cpu")
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        polygons = refiner.refine_boxes(frame, [(10, 20, 60, 90)])

        assert polygons == [[(10, 20), (60, 20), (60, 90), (10, 90)]]

    def test_mask_below_min_area_falls_back_to_box_rectangle(self, fake_sam3):
        fake_sam3.masks_by_box[(10, 20, 60, 90)] = rect_mask((30, 35), (20, 25))  # 25 px
        refiner = Sam3ImageRefiner(device="cpu", min_mask_area=100)
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        polygons = refiner.refine_boxes(frame, [(10, 20, 60, 90)])

        assert polygons == [[(10, 20), (60, 20), (60, 90), (10, 90)]]

    def test_returns_one_polygon_per_box_in_input_order(self, fake_sam3):
        fake_sam3.masks_by_box[(0, 0, 50, 50)] = rect_mask((5, 45), (5, 45))
        fake_sam3.masks_by_box[(100, 100, 150, 180)] = rect_mask((110, 170), (110, 140))
        boxes = [(0, 0, 50, 50), (60, 60, 90, 90), (100, 100, 150, 180)]  # middle: empty mask
        refiner = Sam3ImageRefiner(device="cpu")
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        polygons = refiner.refine_boxes(frame, boxes)

        assert len(polygons) == 3
        assert set(polygons[0]) == rect_corners(5, 5, 44, 44)
        assert polygons[1] == [(60, 60), (90, 60), (90, 90), (60, 90)]
        assert set(polygons[2]) == rect_corners(110, 110, 139, 169)

    def test_no_boxes_returns_empty_list(self, fake_sam3):
        refiner = Sam3ImageRefiner(device="cpu")
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        assert refiner.refine_boxes(frame, []) == []

    def test_predictor_receives_rgb_image_and_float_box_without_multimask(self, fake_sam3):
        refiner = Sam3ImageRefiner(device="cpu")
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[..., 0] = 10  # B
        frame[..., 2] = 200  # R

        refiner.refine_boxes(frame, [(10, 20, 60, 90)])

        predictor = fake_sam3.image_predictors[0]
        assert predictor.images[0][0, 0].tolist() == [200, 0, 10]
        box, multimask = predictor.predict_calls[0]
        assert box.dtype == torch.float32
        assert box.tolist() == [[10.0, 20.0, 60.0, 90.0]]
        assert multimask is False

    def test_predictor_loaded_once_across_calls(self, fake_sam3):
        refiner = Sam3ImageRefiner(model_path="ckpt.pt", device="cpu")
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        refiner.refine_boxes(frame, [(10, 20, 60, 90)])
        refiner.refine_boxes(frame, [(10, 20, 60, 90)])

        assert len(fake_sam3.models) == 1
        assert fake_sam3.models[0].path == "ckpt.pt"
        assert fake_sam3.models[0].device == "cpu"
        assert fake_sam3.models[0].eval_called is True
        assert len(fake_sam3.image_predictors[0].images) == 2

    def test_cuda_device_falls_back_to_cpu_when_cuda_unavailable(self, fake_sam3):
        refiner = Sam3ImageRefiner(device="cuda")
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        refiner.refine_boxes(frame, [(10, 20, 60, 90)])

        assert refiner.device == "cpu"
        assert fake_sam3.models[0].device == "cpu"

    def test_release_drops_predictor_and_reloads_on_next_call(self, fake_sam3):
        refiner = Sam3ImageRefiner(device="cpu")
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        refiner.refine_boxes(frame, [(10, 20, 60, 90)])

        refiner.release()

        assert refiner._predictor is None
        refiner.refine_boxes(frame, [(10, 20, 60, 90)])
        assert len(fake_sam3.models) == 2


# ─── Sam3VideoDetector.detect_video ────────────────────────────


def _config(**overrides):
    base = {"enable_adaptive_intensity": False, "anonymization_intensity": 10}
    base.update(overrides)
    return PipelineConfig(**base)


def _frame_masks(*masks):
    if not masks:
        return np.zeros((0, H, W), dtype=bool)
    return np.stack(masks)


class TestDetectVideo:
    def test_annotations_and_report_per_frame(self, fake_sam3, capsys):
        # Arrange: frame 0 has a large person and a tiny blob (below min area),
        # frame 1 has an empty mask, frame 2 has one person.
        person_a = rect_mask((20, 120), (30, 70))  # contour 39x99
        tiny = rect_mask((150, 158), (200, 208))  # contour 7x7 = 49 px < 100
        person_b = rect_mask((40, 100), (150, 200))  # contour 49x59
        empty = np.zeros((H, W), dtype=bool)
        fake_sam3.video_frames = [
            (0, _frame_masks(person_a, tiny)),
            (1, _frame_masks(empty)),
            (2, _frame_masks(person_b)),
        ]
        detector = Sam3VideoDetector(device="cpu", text_prompt="pedestrian")

        # Act
        annotations, report = detector.detect_video("clip.mp4", _config())

        # Assert: annotations
        assert sorted(annotations) == [0, 1, 2]
        assert [set(p) for p in annotations[0]["auto"]] == [rect_corners(30, 20, 69, 119)]
        assert annotations[0]["intensities"] == [10]
        assert annotations[1] == {"auto": [], "manual": [], "intensities": []}
        assert [set(p) for p in annotations[2]["auto"]] == [rect_corners(150, 40, 199, 99)]
        assert annotations[2]["manual"] == []
        # Assert: report
        assert report[0]["frame_number"] == 0
        assert [report[i]["persons_detected"] for i in (0, 1, 2)] == [1, 0, 1]
        assert report[2] == {
            "frame_number": 2,
            "persons_detected": 1,
            "avg_confidence": 1.0,
            "min_confidence": 1.0,
            "max_confidence": 1.0,
            "motion_zones": 0,
            "sliding_window_hits": 0,
            "multiscale_hits": 0,
            "post_check_alerts": 0,
        }
        # Assert: predictor wiring and key stdout lines
        predictor = fake_sam3.video_predictors[0]
        assert predictor.init_path == "clip.mp4"
        assert predictor.prompt == "pedestrian"
        assert predictor.reset_with == {"state_for": "clip.mp4"}
        out = capsys.readouterr().out
        assert "[FASE 1/5] Rilevamento SAM3 video..." in out
        assert "Istanze totali rilevate:        2" in out
        assert "Frame con 0 rilevamenti:        1" in out

    def test_adaptive_intensity_uses_polygon_height(self, fake_sam3):
        # Polygon spans y 20..119 -> height 99: adaptive = int(10 * 0.99) = 9,
        # floor = max(3, 99 // 4) = 24, so the floor wins.
        fake_sam3.video_frames = [(0, _frame_masks(rect_mask((20, 120), (30, 70))))]
        detector = Sam3VideoDetector(device="cpu")
        config = _config(
            enable_adaptive_intensity=True,
            anonymization_intensity=10,
            adaptive_reference_height=100,
        )

        annotations, _ = detector.detect_video("clip.mp4", config)

        assert annotations[0]["intensities"] == [24]

    def test_two_people_in_one_frame_yield_two_polygons_and_intensities(self, fake_sam3):
        fake_sam3.video_frames = [
            (
                5,
                _frame_masks(rect_mask((20, 120), (30, 70)), rect_mask((40, 100), (150, 200))),
            )
        ]
        detector = Sam3VideoDetector(device="cpu")

        annotations, report = detector.detect_video("clip.mp4", _config())

        assert list(annotations) == [5]  # keyed by the predictor's frame index
        assert len(annotations[5]["auto"]) == 2
        assert annotations[5]["intensities"] == [10, 10]
        assert report[5]["persons_detected"] == 2

    def test_mask_below_min_mask_area_is_dropped(self, fake_sam3):
        blob = rect_mask((20, 40), (30, 50))  # contour 19x19 = 361 px
        fake_sam3.video_frames = [(0, _frame_masks(blob))]

        kept, _ = Sam3VideoDetector(device="cpu", min_mask_area=100).detect_video(
            "clip.mp4", _config()
        )
        dropped, _ = Sam3VideoDetector(device="cpu", min_mask_area=1000).detect_video(
            "clip.mp4", _config()
        )

        assert len(kept[0]["auto"]) == 1
        assert dropped[0]["auto"] == []

    def test_stop_event_already_set_returns_empty_results(self, fake_sam3, capsys):
        fake_sam3.video_frames = [(0, _frame_masks(rect_mask((20, 120), (30, 70))))]
        stop = threading.Event()
        stop.set()
        detector = Sam3VideoDetector(device="cpu")

        annotations, report = detector.detect_video("clip.mp4", _config(), stop_event=stop)

        assert annotations == {}
        assert report == {}
        assert "Pipeline interrotta dall'utente." in capsys.readouterr().out
        assert fake_sam3.video_predictors[0].reset_with == {"state_for": "clip.mp4"}

    def test_stop_event_set_midway_keeps_frames_processed_before_it(self, fake_sam3, capsys):
        person = rect_mask((20, 120), (30, 70))
        fake_sam3.video_frames = [
            (0, _frame_masks(person)),
            (1, _frame_masks(person)),
            (2, _frame_masks(person)),
        ]
        stop = threading.Event()
        detector = Sam3VideoDetector(device="cpu")
        # The fake predictor sets the event right before yielding its second frame.
        original = fake_sam3.SAM3VideoPredictor

        def make_with_stop(model):
            predictor = original(model)
            predictor.stop_after_first = stop
            return predictor

        fake_sam3.SAM3VideoPredictor = make_with_stop

        annotations, report = detector.detect_video("clip.mp4", _config(), stop_event=stop)

        assert sorted(annotations) == [0]
        assert sorted(report) == [0]
        assert fake_sam3.video_predictors[0].yielded == 2  # frame 1 was pulled, then dropped
        assert "Pipeline interrotta dall'utente." in capsys.readouterr().out

    def test_no_frames_returns_empty_results_and_zero_totals(self, fake_sam3, capsys):
        fake_sam3.video_frames = []

        annotations, report = Sam3VideoDetector(device="cpu").detect_video("clip.mp4", _config())

        assert (annotations, report) == ({}, {})
        out = capsys.readouterr().out
        assert "Istanze totali rilevate:        0" in out
        assert "Frame con 0 rilevamenti:        0" in out

    def test_cuda_device_falls_back_to_cpu_when_cuda_unavailable(self, fake_sam3):
        detector = Sam3VideoDetector(device="cuda", model_path="ckpt.pt")

        detector.detect_video("clip.mp4", _config())

        assert detector.device == "cpu"
        assert fake_sam3.models[0].path == "ckpt.pt"
        assert fake_sam3.models[0].device == "cpu"
        assert fake_sam3.models[0].eval_called is True
