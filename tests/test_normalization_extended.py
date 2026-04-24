"""
Test aggiuntivi per normalization.py — coverage righe mancanti.

Copre: _merge_overlapping_rects warning per n>100 (riga 91),
normalize_annotations con poligoni reali (righe 157-181).
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from person_anonymizer.config import PipelineConfig
from person_anonymizer.normalization import _merge_overlapping_rects, normalize_annotations


# ============================================================
# _merge_overlapping_rects — riga 91 (warning n > 100)
# ============================================================


class TestMergeOverlappingRectsLargeInput:
    """Verifica il warning per input grande e il comportamento con n>100 rettangoli."""

    def test_more_than_100_rects_logs_warning(self, caplog):
        # Arrange — 101 rettangoli ben separati (nessun merge)
        rects = [(i * 20, 0, 10, 10) for i in range(101)]

        # Act
        with caplog.at_level(logging.WARNING, logger="person_anonymizer.normalization"):
            result = _merge_overlapping_rects(rects)

        # Assert — warning loggato
        assert any("potenzialmente lenta" in msg or "performance" in msg.lower()
                   for msg in caplog.messages)

    def test_more_than_100_rects_returns_correct_count(self):
        # Arrange — 110 rettangoli non sovrapposti
        rects = [(i * 20, 0, 10, 10) for i in range(110)]

        # Act
        result = _merge_overlapping_rects(rects)

        # Assert — tutti i rettangoli restano (nessun merge)
        assert len(result) == 110

    def test_exactly_100_rects_no_warning(self, caplog):
        # Arrange — esattamente 100 rettangoli (soglia non superata)
        rects = [(i * 20, 0, 10, 10) for i in range(100)]

        # Act
        with caplog.at_level(logging.WARNING, logger="person_anonymizer.normalization"):
            result = _merge_overlapping_rects(rects)

        # Assert — nessun warning per n=100
        warning_msgs = [m for m in caplog.messages if "potenzialmente" in m or "performance" in m.lower()]
        assert len(warning_msgs) == 0


# ============================================================
# normalize_annotations — righe 157-181 (con poligoni reali)
# ============================================================


class TestNormalizeAnnotationsWithPolygons:
    """Verifica normalize_annotations con poligoni reali (richiede cv2.boundingRect)."""

    def test_single_rectangular_polygon_normalized(self):
        # Arrange — un poligono rettangolare [(10,10),(50,10),(50,50),(10,50)]
        config = PipelineConfig(
            enable_adaptive_intensity=False,
            anonymization_intensity=15,
        )
        annotations = {
            0: {
                "auto": [[(10, 10), (50, 10), (50, 50), (10, 50)]],
                "manual": [],
            }
        }

        # Act
        result, (before, after) = normalize_annotations(annotations, config)

        # Assert — 1 poligono → 1 rettangolo normalizzato
        assert before == 1
        assert after == 1
        assert len(result[0]["auto"]) == 1
        assert result[0]["manual"] == []

    def test_two_overlapping_polygons_merged(self):
        # Arrange — 2 poligoni sovrapposti → devono essere unificati in 1
        config = PipelineConfig(
            enable_adaptive_intensity=False,
            anonymization_intensity=10,
        )
        # Poligono A: (0,0)-(40,40); Poligono B: (20,20)-(60,60) → si sovrappongono
        annotations = {
            0: {
                "auto": [
                    [(0, 0), (40, 0), (40, 40), (0, 40)],
                    [(20, 20), (60, 20), (60, 60), (20, 60)],
                ],
                "manual": [],
            }
        }

        # Act
        result, (before, after) = normalize_annotations(annotations, config)

        # Assert — 2 poligoni sovrapposti → 1 rettangolo unificato
        assert before == 2
        assert after == 1
        assert len(result[0]["auto"]) == 1

    def test_two_separate_polygons_stay_separate(self):
        # Arrange — 2 poligoni distanti → nessun merge
        config = PipelineConfig(
            enable_adaptive_intensity=False,
            anonymization_intensity=10,
        )
        annotations = {
            0: {
                "auto": [
                    [(0, 0), (20, 0), (20, 20), (0, 20)],
                    [(100, 100), (120, 100), (120, 120), (100, 120)],
                ],
                "manual": [],
            }
        }

        # Act
        result, (before, after) = normalize_annotations(annotations, config)

        # Assert — 2 poligoni separati rimangono 2
        assert before == 2
        assert after == 2
        assert len(result[0]["auto"]) == 2

    def test_polygon_converted_to_rectangle(self):
        # Arrange — un poligono normalizzato deve diventare lista di 4 punti
        config = PipelineConfig(enable_adaptive_intensity=False, anonymization_intensity=10)
        annotations = {
            0: {
                "auto": [[(5, 5), (30, 5), (30, 40), (5, 40)]],
                "manual": [],
            }
        }

        # Act
        result, _ = normalize_annotations(annotations, config)

        # Assert — il poligono output ha 4 punti
        poly = result[0]["auto"][0]
        assert len(poly) == 4

    def test_intensities_computed_for_each_polygon(self):
        # Arrange — adaptive_intensity disabilitata → intensità fissa
        config = PipelineConfig(
            enable_adaptive_intensity=False,
            anonymization_intensity=20,
        )
        annotations = {
            0: {
                "auto": [
                    [(0, 0), (30, 0), (30, 30), (0, 30)],
                    [(50, 50), (80, 50), (80, 80), (50, 80)],
                ],
                "manual": [],
            }
        }

        # Act
        result, _ = normalize_annotations(annotations, config)

        # Assert — intensità per ogni poligono = anonymization_intensity
        intensities = result[0]["intensities"]
        assert len(intensities) == len(result[0]["auto"])
        for intensity in intensities:
            assert intensity == 20

    def test_adaptive_intensity_enabled(self):
        # Arrange — adaptive intensity abilitata → intensità proporzionale all'altezza
        config = PipelineConfig(
            enable_adaptive_intensity=True,
            anonymization_intensity=10,
            adaptive_reference_height=80,
        )
        # Poligono alto 40px (metà della reference)
        annotations = {
            0: {
                "auto": [[(0, 0), (50, 0), (50, 40), (0, 40)]],
                "manual": [],
            }
        }

        # Act
        result, _ = normalize_annotations(annotations, config)

        # Assert — intensità calcolata (non uguale a quella di base)
        intensities = result[0]["intensities"]
        assert len(intensities) == 1
        assert intensities[0] > 0

    def test_manual_polygons_merged_with_auto(self):
        # Arrange — manual polygon viene unito all'auto in normalizzazione
        config = PipelineConfig(enable_adaptive_intensity=False, anonymization_intensity=10)
        annotations = {
            0: {
                "auto": [[(0, 0), (20, 0), (20, 20), (0, 20)]],
                "manual": [[(50, 50), (70, 50), (70, 70), (50, 70)]],
            }
        }

        # Act
        result, (before, after) = normalize_annotations(annotations, config)

        # Assert — auto + manual = 2 polys before; output ha manual=[] (tutti in auto)
        assert before == 2
        assert result[0]["manual"] == []

    def test_multiple_frames_processed_independently(self):
        # Arrange — 3 frame, ognuno con 1 poligono
        config = PipelineConfig(enable_adaptive_intensity=False, anonymization_intensity=10)
        poly = [(0, 0), (20, 0), (20, 20), (0, 20)]
        annotations = {
            0: {"auto": [poly], "manual": []},
            1: {"auto": [poly], "manual": []},
            5: {"auto": [poly], "manual": []},
        }

        # Act
        result, (before, after) = normalize_annotations(annotations, config)

        # Assert
        assert before == 3
        assert after == 3
        assert len(result) == 3
