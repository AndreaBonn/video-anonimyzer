"""
Test per tracking.py — TemporalSmoother con EMA e ghost boxes.

Non richiede cv2, ultralytics o modelli pesanti: TemporalSmoother
usa solo numpy internamente.
"""

import pytest

from tracking import TemporalSmoother

# ============================================================
# TemporalSmoother.smooth
# ============================================================


class TestSmooth:
    """Verifica l'applicazione dell'EMA e la gestione dello stato per track."""

    def test_new_track_returns_same_coords(self):
        # Arrange
        smoother = TemporalSmoother(alpha=0.5)

        # Act — primo smooth: lo stato è vuoto, le coordinate vengono salvate direttamente
        result = smoother.smooth(track_id=1, x1=10, y1=20, x2=100, y2=200)

        # Assert — deve restituire le stesse coordinate passate, castato a int
        assert result == (10, 20, 100, 200)

    def test_existing_track_applies_ema(self):
        # Arrange — alpha = 1.0 significa che il risultato è sempre la posizione corrente
        alpha = 0.4
        smoother = TemporalSmoother(alpha=alpha)
        # Primo smooth: inizializza lo stato
        smoother.smooth(track_id=1, x1=0, y1=0, x2=100, y2=100)

        # Act — secondo smooth: applica EMA
        result = smoother.smooth(track_id=1, x1=50, y1=50, x2=150, y2=150)

        # Assert — formula EMA: alpha * current + (1-alpha) * previous
        # prev = [0, 0, 100, 100], curr = [50, 50, 150, 150]
        expected_x1 = int(alpha * 50 + (1 - alpha) * 0)  # 20
        expected_y1 = int(alpha * 50 + (1 - alpha) * 0)  # 20
        expected_x2 = int(alpha * 150 + (1 - alpha) * 100)  # 120
        expected_y2 = int(alpha * 150 + (1 - alpha) * 100)  # 120
        assert result == (expected_x1, expected_y1, expected_x2, expected_y2)

    def test_smooth_resets_ghost_countdown(self):
        # Arrange — simula un track che entra in ghost e poi riappare
        smoother = TemporalSmoother(alpha=0.5, ghost_frames=5)
        smoother.smooth(track_id=1, x1=0, y1=0, x2=50, y2=50)
        # Mette il track in ghost
        smoother.clear_stale(active_ids=set())
        assert 1 in smoother.ghost_countdown

        # Act — il track riappare
        smoother.smooth(track_id=1, x1=10, y1=10, x2=60, y2=60)

        # Assert — il countdown ghost deve essere rimosso
        assert 1 not in smoother.ghost_countdown


# ============================================================
# TemporalSmoother.get_ghost_boxes
# ============================================================


class TestGetGhostBoxes:
    """Verifica la generazione e il ciclo di vita dei ghost box."""

    def test_no_ghosts_initially(self):
        # Arrange
        smoother = TemporalSmoother(alpha=0.5)

        # Act / Assert — nessun ghost box all'inizializzazione
        assert smoother.get_ghost_boxes() == []

    def test_ghost_box_after_stale(self):
        # Arrange — registra un track e rendilo stale
        smoother = TemporalSmoother(alpha=0.5, ghost_frames=3)
        smoother.smooth(track_id=7, x1=10, y1=10, x2=50, y2=50)
        smoother.clear_stale(active_ids=set())

        # Act
        ghosts = smoother.get_ghost_boxes()

        # Assert — deve essere presente un ghost box per il track 7
        assert len(ghosts) == 1
        assert ghosts[0][0] == 7

    def test_ghost_countdown_decrements(self):
        # Arrange
        smoother = TemporalSmoother(alpha=0.5, ghost_frames=5)
        smoother.smooth(track_id=2, x1=0, y1=0, x2=100, y2=100)
        smoother.clear_stale(active_ids=set())
        assert smoother.ghost_countdown[2] == 5

        # Act — ogni chiamata a get_ghost_boxes deve decrementare il countdown
        smoother.get_ghost_boxes()

        # Assert — countdown sceso da 5 a 4
        assert smoother.ghost_countdown[2] == 4

    def test_ghost_expansion_applied(self):
        # Arrange — track centrato in (100, 100), larghezza 100, altezza 100
        # ghost_expansion = 1.5 → w e h espansi del 50%
        smoother = TemporalSmoother(alpha=1.0, ghost_frames=2, ghost_expansion=1.5)
        smoother.smooth(track_id=3, x1=50, y1=50, x2=150, y2=150)
        smoother.clear_stale(active_ids=set())

        # Act
        ghosts = smoother.get_ghost_boxes()

        # Assert — centro (100, 100), w=100*1.5=150, h=100*1.5=150
        # x1_ghost = 100 - 75 = 25, x2_ghost = 100 + 75 = 175
        assert len(ghosts) == 1
        _, gx1, gy1, gx2, gy2 = ghosts[0]
        # L'espansione centra il box: verifica che sia più grande dell'originale
        orig_w = 150 - 50  # 100
        ghost_w = gx2 - gx1
        assert ghost_w > orig_w
        # Verifica valori precisi (alpha=1.0 → stato = coords esatte)
        assert gx1 == 25
        assert gy1 == 25
        assert gx2 == 175
        assert gy2 == 175

    def test_ghost_eventually_removed(self):
        # Arrange — un ghost deve scomparire dopo un numero finito di chiamate,
        # indipendentemente dai dettagli di implementazione del countdown
        smoother = TemporalSmoother(alpha=0.5, ghost_frames=3)
        smoother.smooth(track_id=4, x1=0, y1=0, x2=40, y2=40)
        smoother.clear_stale(active_ids=set())

        # Act — chiama get_ghost_boxes abbondantemente oltre ghost_frames
        for _ in range(10):
            smoother.get_ghost_boxes()

        # Assert — il track deve essere rimosso: nessun ghost box residuo
        # e nessuna traccia nello stato interno
        assert smoother.get_ghost_boxes() == []
        assert 4 not in smoother.state

    def test_ghost_persists_during_countdown(self):
        # Arrange — un ghost con 5 frame deve essere ancora visibile dopo 2 chiamate
        smoother = TemporalSmoother(alpha=0.5, ghost_frames=5)
        smoother.smooth(track_id=5, x1=10, y1=10, x2=60, y2=60)
        smoother.clear_stale(active_ids=set())

        # Act — solo 2 chiamate su 5: il ghost deve essere ancora presente
        smoother.get_ghost_boxes()
        ghosts = smoother.get_ghost_boxes()

        # Assert — il ghost deve ancora esistere
        assert len(ghosts) == 1
        assert ghosts[0][0] == 5


# ============================================================
# TemporalSmoother.clear_stale
# ============================================================


class TestClearStale:
    """Verifica la marcatura dei track inattivi per il ghost countdown."""

    def test_marks_inactive_tracks_for_ghost(self):
        # Arrange — due track registrati, nessuno nella lista active
        smoother = TemporalSmoother(alpha=0.5, ghost_frames=8)
        smoother.smooth(track_id=10, x1=0, y1=0, x2=20, y2=20)
        smoother.smooth(track_id=11, x1=50, y1=50, x2=80, y2=80)

        # Act — nessun track attivo: entrambi devono entrare in ghost
        smoother.clear_stale(active_ids=set())

        # Assert
        assert 10 in smoother.ghost_countdown
        assert smoother.ghost_countdown[10] == 8
        assert 11 in smoother.ghost_countdown
        assert smoother.ghost_countdown[11] == 8

    def test_active_tracks_not_marked(self):
        # Arrange — due track registrati, solo il primo è ancora attivo
        smoother = TemporalSmoother(alpha=0.5, ghost_frames=5)
        smoother.smooth(track_id=20, x1=0, y1=0, x2=30, y2=30)
        smoother.smooth(track_id=21, x1=60, y1=60, x2=90, y2=90)

        # Act — solo track 20 è attivo
        smoother.clear_stale(active_ids={20})

        # Assert — track 20 non deve avere ghost countdown; track 21 sì
        assert 20 not in smoother.ghost_countdown
        assert 21 in smoother.ghost_countdown
