"""
Test per postprocessing.py — funzioni pure senza cv2, ffmpeg o YOLO.

_rects_overlap, _merge_rects, _merge_overlapping_rects e
normalize_annotations (solo path senza poligoni, per evitare cv2.boundingRect)
sono testate in isolamento tramite import diretto.
"""

import pytest

from config import PipelineConfig
from postprocessing import (
    _merge_overlapping_rects,
    _merge_rects,
    _rects_overlap,
    filter_artifact_detections,
    normalize_annotations,
)

# ============================================================
# _rects_overlap
# ============================================================


class TestRectsOverlap:
    """Verifica la rilevazione di sovrapposizione tra rettangoli (x, y, w, h)."""

    def test_overlapping_true(self):
        # Arrange — due rettangoli che si sovrappongono al centro
        r1 = (0, 0, 50, 50)
        r2 = (25, 25, 50, 50)

        # Act
        result = _rects_overlap(r1, r2)

        # Assert
        assert result is True

    def test_separated_false(self):
        # Arrange — due rettangoli distanti senza alcun contatto
        r1 = (0, 0, 10, 10)
        r2 = (50, 50, 10, 10)

        # Act
        result = _rects_overlap(r1, r2)

        # Assert
        assert result is False

    def test_touching_edges_false(self):
        # Arrange — r1 finisce a x=10 (0+10), r2 inizia a x=10 → adiacenti, non sovrapposti
        r1 = (0, 0, 10, 10)
        r2 = (10, 0, 10, 10)

        # Act
        result = _rects_overlap(r1, r2)

        # Assert — bordi a contatto non costituiscono overlap
        assert result is False

    def test_touching_top_bottom_false(self):
        # Arrange — r1 finisce a y=10, r2 inizia a y=10
        r1 = (0, 0, 10, 10)
        r2 = (0, 10, 10, 10)

        # Act
        result = _rects_overlap(r1, r2)

        # Assert
        assert result is False

    def test_contained_rect_true(self):
        # Arrange — r2 completamente dentro r1
        r1 = (0, 0, 100, 100)
        r2 = (10, 10, 30, 30)

        # Act
        result = _rects_overlap(r1, r2)

        # Assert
        assert result is True

    def test_symmetry(self):
        # Arrange
        r1 = (0, 0, 30, 30)
        r2 = (20, 20, 30, 30)

        # Act / Assert — _rects_overlap deve essere simmetrica
        assert _rects_overlap(r1, r2) == _rects_overlap(r2, r1)


# ============================================================
# _merge_rects
# ============================================================


class TestMergeRects:
    """Verifica il merge di due rettangoli nel bounding box comune."""

    def test_merge_two_rects(self):
        # Arrange
        r1 = (0, 0, 10, 10)
        r2 = (5, 5, 10, 10)

        # Act
        result = _merge_rects(r1, r2)

        # Assert — bounding box che contiene entrambi
        x, y, w, h = result
        assert x == 0
        assert y == 0
        assert x + w == 15  # max(0+10, 5+10)
        assert y + h == 15

    def test_merge_separated_rects(self):
        # Arrange
        r1 = (0, 0, 10, 10)
        r2 = (20, 20, 10, 10)

        # Act
        x, y, w, h = _merge_rects(r1, r2)

        # Assert — bbox che racchiude entrambi
        assert x == 0
        assert y == 0
        assert x + w == 30
        assert y + h == 30

    def test_merge_identical_rects(self):
        # Arrange
        r = (5, 5, 20, 20)

        # Act
        result = _merge_rects(r, r)

        # Assert
        assert result == r

    def test_merge_result_contains_both(self):
        # Arrange
        r1 = (3, 7, 15, 20)
        r2 = (10, 2, 8, 30)

        # Act
        rx, ry, rw, rh = _merge_rects(r1, r2)

        # Assert — il rettangolo risultante contiene entrambi
        assert rx <= r1[0] and rx <= r2[0]
        assert ry <= r1[1] and ry <= r2[1]
        assert rx + rw >= r1[0] + r1[2]
        assert rx + rw >= r2[0] + r2[2]
        assert ry + rh >= r1[1] + r1[3]
        assert ry + rh >= r2[1] + r2[3]


# ============================================================
# _merge_overlapping_rects
# ============================================================


class TestMergeOverlappingRects:
    """Verifica il merge iterativo di rettangoli sovrapposti."""

    def test_no_overlap_returns_same_count(self):
        # Arrange — 3 rettangoli ben separati
        rects = [
            (0, 0, 10, 10),
            (50, 0, 10, 10),
            (100, 0, 10, 10),
        ]

        # Act
        result = _merge_overlapping_rects(rects)

        # Assert — nessun merge, 3 rettangoli restano
        assert len(result) == 3

    def test_all_overlapping_returns_one(self):
        # Arrange — 3 rettangoli che si sovrappongono tutti
        rects = [
            (0, 0, 30, 30),
            (10, 10, 30, 30),
            (20, 20, 30, 30),
        ]

        # Act
        result = _merge_overlapping_rects(rects)

        # Assert — un solo rettangolo risultante
        assert len(result) == 1

    def test_chain_overlap_transitive_merge(self):
        # Arrange — A sovrappone B, B sovrappone C, A NON sovrappone C direttamente
        # A: [0,0,10,10] → x in [0,10)
        # B: [8,0,10,10] → x in [8,18)
        # C: [16,0,10,10] → x in [16,26)
        # A∩B: overlap (8<10 e 10>8 → True); B∩C: overlap; A∩C: A finisce a 10, C inizia a 16 → no overlap
        rects = [
            (0, 0, 10, 10),
            (8, 0, 10, 10),
            (16, 0, 10, 10),
        ]

        # Act
        result = _merge_overlapping_rects(rects)

        # Assert — merge transitivo: tutti uniti in 1
        assert len(result) == 1

    def test_empty_input(self):
        # Arrange / Act
        result = _merge_overlapping_rects([])

        # Assert
        assert result == []

    def test_single_rect_unchanged(self):
        # Arrange
        rects = [(5, 5, 20, 20)]

        # Act
        result = _merge_overlapping_rects(rects)

        # Assert
        assert len(result) == 1
        assert result[0] == rects[0]

    def test_merged_rect_contains_all_input(self):
        # Arrange
        rects = [
            (0, 0, 20, 20),
            (10, 10, 20, 20),
            (5, 15, 20, 10),
        ]

        # Act
        result = _merge_overlapping_rects(rects)

        # Assert — il risultato contiene tutti i rettangoli originali
        assert len(result) == 1
        rx, ry, rw, rh = result[0]
        for x, y, w, h in rects:
            assert rx <= x
            assert ry <= y
            assert rx + rw >= x + w
            assert ry + rh >= y + h


# ============================================================
# normalize_annotations
# ============================================================


class TestNormalizeAnnotations:
    """Verifica la normalizzazione delle annotazioni."""

    def test_empty_annotations(self):
        # Arrange
        config = PipelineConfig()
        annotations = {}

        # Act
        result, (before, after) = normalize_annotations(annotations, config)

        # Assert
        assert result == {}
        assert before == 0
        assert after == 0

    def test_frame_with_no_polygons(self):
        # Arrange — frame presente ma senza poligoni
        config = PipelineConfig()
        annotations = {
            0: {"auto": [], "manual": [], "intensities": []},
        }

        # Act
        result, (before, after) = normalize_annotations(annotations, config)

        # Assert
        assert 0 in result
        assert result[0]["auto"] == []
        assert result[0]["manual"] == []
        assert before == 0
        assert after == 0

    def test_multiple_empty_frames(self):
        # Arrange
        config = PipelineConfig()
        annotations = {
            0: {"auto": [], "manual": []},
            1: {"auto": [], "manual": []},
            5: {"auto": [], "manual": []},
        }

        # Act
        result, (before, after) = normalize_annotations(annotations, config)

        # Assert
        assert len(result) == 3
        assert before == 0
        assert after == 0

    def test_output_structure_keys(self):
        # Arrange
        config = PipelineConfig()
        annotations = {0: {"auto": [], "manual": []}}

        # Act
        result, _ = normalize_annotations(annotations, config)

        # Assert — ogni frame ha le chiavi attese
        assert "auto" in result[0]
        assert "manual" in result[0]
        assert "intensities" in result[0]

    def test_intensities_disabled(self):
        # Arrange — adaptive intensity disabilitata → intensità fissa
        config = PipelineConfig(
            enable_adaptive_intensity=False,
            anonymization_intensity=15,
        )
        annotations = {0: {"auto": [], "manual": []}}

        # Act
        result, _ = normalize_annotations(annotations, config)

        # Assert — nessun poligono → nessuna intensità
        assert result[0]["intensities"] == []


# ============================================================
# filter_artifact_detections
# ============================================================


class TestFilterArtifactDetections:
    """Verifica il filtraggio di rilevamenti artefatto post-render.

    La funzione confronta ogni detection box (x1, y1, x2, y2, conf) con le
    annotazioni esistenti del frame: se IoU >= threshold il box è artefatto,
    altrimenti è genuino.

    I poligoni di annotazione sono rettangolari per semplificare il calcolo
    manuale dell'IoU atteso. polygon_to_bbox estrae il bounding box minimo,
    quindi [(x1,y1),(x2,y1),(x2,y2),(x1,y2)] produce bbox [x1, y1, x2, y2].
    """

    def test_empty_alert_frames(self):
        # Arrange — nessun frame di alert
        alert_frames = []
        annotations = {}
        iou_threshold = 0.5

        # Act
        genuine_alerts, total_artifacts, total_genuine = filter_artifact_detections(
            alert_frames, annotations, iou_threshold
        )

        # Assert
        assert genuine_alerts == []
        assert total_artifacts == 0
        assert total_genuine == 0

    def test_all_artifacts_filtered(self):
        # Arrange — detection box identico al poligono annotato: IoU = 1.0 >= 0.5
        # Annotazione sul frame 0: poligono rettangolare [10,10] → [50,50]
        # Detection box uguale: [10, 10, 50, 50, 0.9] → IoU = 1.0
        annotations = {
            0: {
                "auto": [[(10, 10), (50, 10), (50, 50), (10, 50)]],
                "manual": [],
            }
        }
        nms_boxes = [[10, 10, 50, 50, 0.9]]
        alert_frames = [(0, 1, nms_boxes)]
        iou_threshold = 0.5

        # Act
        genuine_alerts, total_artifacts, total_genuine = filter_artifact_detections(
            alert_frames, annotations, iou_threshold
        )

        # Assert — nessun box genuino, un artefatto
        assert genuine_alerts == []
        assert total_artifacts == 1
        assert total_genuine == 0

    def test_genuine_detection_kept(self):
        # Arrange — detection box in zona non annotata: IoU = 0.0 < 0.5
        # Annotazione: [0, 0, 40, 40]; detection lontana: [100, 100, 140, 140]
        annotations = {
            0: {
                "auto": [[(0, 0), (40, 0), (40, 40), (0, 40)]],
                "manual": [],
            }
        }
        nms_boxes = [[100, 100, 140, 140, 0.85]]
        alert_frames = [(0, 1, nms_boxes)]
        iou_threshold = 0.5

        # Act
        genuine_alerts, total_artifacts, total_genuine = filter_artifact_detections(
            alert_frames, annotations, iou_threshold
        )

        # Assert — il box è genuino: deve comparire in genuine_alerts
        assert len(genuine_alerts) == 1
        frame_idx, genuine_boxes = genuine_alerts[0]
        assert frame_idx == 0
        assert len(genuine_boxes) == 1
        assert genuine_boxes[0] == nms_boxes[0]
        assert total_artifacts == 0
        assert total_genuine == 1

    def test_mixed_artifacts_and_genuine(self):
        # Arrange — due detection box sullo stesso frame:
        #   box A [0, 0, 40, 40] sovrapposto con annotazione [0,0,40,40] → IoU=1.0 → artefatto
        #   box B [200, 200, 240, 240] lontano da annotazione → IoU=0.0 → genuino
        annotations = {
            5: {
                "auto": [[(0, 0), (40, 0), (40, 40), (0, 40)]],
                "manual": [],
            }
        }
        box_artifact = [0, 0, 40, 40, 0.95]
        box_genuine = [200, 200, 240, 240, 0.80]
        nms_boxes = [box_artifact, box_genuine]
        alert_frames = [(5, 2, nms_boxes)]
        iou_threshold = 0.5

        # Act
        genuine_alerts, total_artifacts, total_genuine = filter_artifact_detections(
            alert_frames, annotations, iou_threshold
        )

        # Assert
        assert total_artifacts == 1
        assert total_genuine == 1
        assert len(genuine_alerts) == 1
        frame_idx, genuine_boxes = genuine_alerts[0]
        assert frame_idx == 5
        assert len(genuine_boxes) == 1
        assert genuine_boxes[0] == box_genuine

    def test_no_existing_annotations(self):
        # Arrange — frame senza alcuna annotazione: nessun bbox esistente
        # → ogni detection ha IoU=0.0 con tutti gli existing (lista vuota)
        # → tutti i box sono genuini
        annotations = {}
        nms_boxes = [
            [10, 10, 50, 50, 0.9],
            [60, 60, 100, 100, 0.75],
        ]
        alert_frames = [(3, 2, nms_boxes)]
        iou_threshold = 0.5

        # Act
        genuine_alerts, total_artifacts, total_genuine = filter_artifact_detections(
            alert_frames, annotations, iou_threshold
        )

        # Assert — tutti genuini, nessun artefatto
        assert total_artifacts == 0
        assert total_genuine == 2
        assert len(genuine_alerts) == 1
        _, genuine_boxes = genuine_alerts[0]
        assert len(genuine_boxes) == 2
