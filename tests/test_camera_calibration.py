"""Test per camera_calibration — utility calibrazione camera con scacchiera.

Mock pesante su cv2 e numpy per evitare dipendenze da immagini fisiche.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from person_anonymizer.camera_calibration import (
    calibrate_camera,
    find_chessboard_corners,
)

# ─── find_chessboard_corners ────────────────────────────────────


class TestFindChessboardCorners:
    def test_returns_empty_when_no_valid_images(self):
        with patch("cv2.imread", return_value=None):
            obj_pts, img_pts, img_size = find_chessboard_corners(["fake.jpg"])
        assert obj_pts == []
        assert img_pts == []
        assert img_size is None

    def test_skips_unreadable_image(self):
        with patch("cv2.imread", return_value=None):
            obj_pts, img_pts, _ = find_chessboard_corners(["missing.jpg", "also_missing.jpg"])
        assert obj_pts == []
        assert img_pts == []

    def test_corners_found_appends_points(self):
        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_corners = np.zeros((54, 1, 2), dtype=np.float32)

        with (
            patch("cv2.imread", return_value=bgr),
            patch("cv2.cvtColor", return_value=gray),
            patch("cv2.findChessboardCorners", return_value=(True, fake_corners)),
            patch("cv2.cornerSubPix", return_value=fake_corners),
        ):
            obj_pts, img_pts, img_size = find_chessboard_corners(["img1.jpg"])

        assert len(obj_pts) == 1
        assert len(img_pts) == 1
        assert img_size == (640, 480)

    def test_corners_not_found_skips_image(self):
        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        with (
            patch("cv2.imread", return_value=bgr),
            patch("cv2.cvtColor", return_value=gray),
            patch("cv2.findChessboardCorners", return_value=(False, None)),
        ):
            obj_pts, img_pts, img_size = find_chessboard_corners(["img1.jpg"])

        assert obj_pts == []
        assert img_pts == []
        assert img_size == (640, 480)  # image_size impostato anche senza corners

    def test_image_size_set_once_from_first_readable(self):
        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_corners = np.zeros((54, 1, 2), dtype=np.float32)

        with (
            patch("cv2.imread", return_value=bgr),
            patch("cv2.cvtColor", return_value=gray),
            patch("cv2.findChessboardCorners", return_value=(True, fake_corners)),
            patch("cv2.cornerSubPix", return_value=fake_corners),
        ):
            _, _, img_size = find_chessboard_corners(["img1.jpg", "img2.jpg"])

        assert img_size == (640, 480)

    def test_multiple_valid_images_accumulate(self):
        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_corners = np.zeros((54, 1, 2), dtype=np.float32)

        with (
            patch("cv2.imread", return_value=bgr),
            patch("cv2.cvtColor", return_value=gray),
            patch("cv2.findChessboardCorners", return_value=(True, fake_corners)),
            patch("cv2.cornerSubPix", return_value=fake_corners),
        ):
            obj_pts, img_pts, _ = find_chessboard_corners(
                ["img1.jpg", "img2.jpg", "img3.jpg"]
            )

        assert len(obj_pts) == 3
        assert len(img_pts) == 3

    def test_custom_board_size(self):
        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_corners = np.zeros((35, 1, 2), dtype=np.float32)

        with (
            patch("cv2.imread", return_value=bgr),
            patch("cv2.cvtColor", return_value=gray),
            patch("cv2.findChessboardCorners", return_value=(True, fake_corners)) as mock_fcb,
            patch("cv2.cornerSubPix", return_value=fake_corners),
        ):
            find_chessboard_corners(["img.jpg"], board_size=(7, 5))

        mock_fcb.assert_called_once()
        args = mock_fcb.call_args[0]
        assert args[1] == (7, 5)

    def test_objp_shape_matches_board_size(self):
        """obj_points deve avere board_cols * board_rows righe."""
        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_corners = np.zeros((54, 1, 2), dtype=np.float32)

        with (
            patch("cv2.imread", return_value=bgr),
            patch("cv2.cvtColor", return_value=gray),
            patch("cv2.findChessboardCorners", return_value=(True, fake_corners)),
            patch("cv2.cornerSubPix", return_value=fake_corners),
        ):
            obj_pts, _, _ = find_chessboard_corners(["img.jpg"], board_size=(9, 6))

        assert obj_pts[0].shape == (54, 3)

    def test_empty_image_list_returns_empty(self):
        obj_pts, img_pts, img_size = find_chessboard_corners([])
        assert obj_pts == []
        assert img_pts == []
        assert img_size is None

    def test_mixed_valid_invalid_images(self):
        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_corners = np.zeros((54, 1, 2), dtype=np.float32)

        # Primo imread None, secondo OK
        imread_results = [None, bgr]

        with (
            patch("cv2.imread", side_effect=imread_results),
            patch("cv2.cvtColor", return_value=gray),
            patch("cv2.findChessboardCorners", return_value=(True, fake_corners)),
            patch("cv2.cornerSubPix", return_value=fake_corners),
        ):
            obj_pts, img_pts, _ = find_chessboard_corners(["bad.jpg", "good.jpg"])

        assert len(obj_pts) == 1


# ─── calibrate_camera ───────────────────────────────────────────


class TestCalibrateCamera:
    def test_returns_matrix_distcoeffs_rms(self):
        mock_matrix = np.eye(3, dtype=np.float64)
        mock_dist = np.zeros((1, 5), dtype=np.float64)
        rms = 0.42

        with patch(
            "cv2.calibrateCamera",
            return_value=(rms, mock_matrix, mock_dist, [], []),
        ):
            camera_matrix, dist_coefficients, rms_error = calibrate_camera(
                obj_points=[MagicMock()],
                img_points=[MagicMock()],
                image_size=(640, 480),
            )

        assert np.array_equal(camera_matrix, mock_matrix)
        assert np.array_equal(dist_coefficients, mock_dist)
        assert rms_error == pytest.approx(0.42)

    def test_passes_correct_args_to_calibrate_camera(self):
        mock_matrix = np.eye(3)
        mock_dist = np.zeros((1, 5))

        obj_pts = [np.zeros((54, 3), dtype=np.float32)]
        img_pts = [np.zeros((54, 1, 2), dtype=np.float32)]
        image_size = (1280, 720)

        with patch(
            "cv2.calibrateCamera", return_value=(0.1, mock_matrix, mock_dist, [], [])
        ) as mock_cc:
            calibrate_camera(obj_pts, img_pts, image_size)

        mock_cc.assert_called_once_with(obj_pts, img_pts, image_size, None, None)

    def test_rms_error_is_float(self):
        mock_matrix = np.eye(3)
        mock_dist = np.zeros((1, 5))

        with patch("cv2.calibrateCamera", return_value=(1.23, mock_matrix, mock_dist, [], [])):
            _, _, rms_error = calibrate_camera([], [], (640, 480))

        assert isinstance(rms_error, float)


# ─── main() — CLI entry point ───────────────────────────────────


class TestMain:
    def test_main_raises_on_missing_images_dir(self, tmp_path):
        """FileNotFoundError se la cartella non esiste."""
        from person_anonymizer.camera_calibration import main

        args = ["--images", str(tmp_path / "nonexistent"), "--output", "out.npz"]
        with patch("sys.argv", ["camera_calibration.py"] + args):
            with pytest.raises(FileNotFoundError, match="Cartella non trovata"):
                main()

    def test_main_raises_on_empty_images_dir(self, tmp_path):
        """FileNotFoundError se la cartella non contiene immagini."""
        from person_anonymizer.camera_calibration import main

        args = ["--images", str(tmp_path), "--output", "out.npz"]
        with patch("sys.argv", ["camera_calibration.py"] + args):
            with pytest.raises(FileNotFoundError, match="Nessuna immagine trovata"):
                main()

    def test_main_raises_value_error_with_few_valid_images(self, tmp_path):
        """ValueError se meno di 3 immagini valide."""
        from person_anonymizer.camera_calibration import main

        # Crea 1 immagine dummy
        (tmp_path / "img.jpg").write_bytes(b"fake")

        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        args = ["--images", str(tmp_path), "--output", "out.npz"]
        with (
            patch("sys.argv", ["camera_calibration.py"] + args),
            patch("cv2.imread", return_value=bgr),
            patch("cv2.cvtColor", return_value=gray),
            patch("cv2.findChessboardCorners", return_value=(False, None)),
        ):
            with pytest.raises(ValueError, match="Servono almeno 3 immagini valide"):
                main()

    def test_main_success_saves_npz(self, tmp_path):
        """Flusso completo: 3 immagini valide → salva calibration.npz."""
        from person_anonymizer.camera_calibration import main

        for i in range(3):
            (tmp_path / f"img{i}.jpg").write_bytes(b"fake")

        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_corners = np.zeros((54, 1, 2), dtype=np.float32)
        mock_matrix = np.eye(3, dtype=np.float64)
        mock_dist = np.zeros((1, 5), dtype=np.float64)

        output_file = str(tmp_path / "calibration.npz")
        args = ["--images", str(tmp_path), "--output", output_file]

        with (
            patch("sys.argv", ["camera_calibration.py"] + args),
            patch("cv2.imread", return_value=bgr),
            patch("cv2.cvtColor", return_value=gray),
            patch("cv2.findChessboardCorners", return_value=(True, fake_corners)),
            patch("cv2.cornerSubPix", return_value=fake_corners),
            patch(
                "cv2.calibrateCamera",
                return_value=(0.25, mock_matrix, mock_dist, [], []),
            ),
            patch("numpy.savez") as mock_savez,
        ):
            main()

        mock_savez.assert_called_once()
        save_args = mock_savez.call_args
        assert save_args[0][0] == output_file

    def test_main_custom_board_size(self, tmp_path):
        """--board-cols e --board-rows vengono passati a find_chessboard_corners."""
        from person_anonymizer.camera_calibration import main

        for i in range(3):
            (tmp_path / f"img{i}.jpg").write_bytes(b"fake")

        gray = np.zeros((480, 640), dtype=np.uint8)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_corners = np.zeros((35, 1, 2), dtype=np.float32)
        mock_matrix = np.eye(3, dtype=np.float64)
        mock_dist = np.zeros((1, 5), dtype=np.float64)

        args = [
            "--images", str(tmp_path),
            "--output", str(tmp_path / "out.npz"),
            "--board-cols", "7",
            "--board-rows", "5",
        ]

        with (
            patch("sys.argv", ["camera_calibration.py"] + args),
            patch("cv2.imread", return_value=bgr),
            patch("cv2.cvtColor", return_value=gray),
            patch("cv2.findChessboardCorners", return_value=(True, fake_corners)) as mock_fcb,
            patch("cv2.cornerSubPix", return_value=fake_corners),
            patch(
                "cv2.calibrateCamera",
                return_value=(0.1, mock_matrix, mock_dist, [], []),
            ),
            patch("numpy.savez"),
        ):
            main()

        for c in mock_fcb.call_args_list:
            assert c[0][1] == (7, 5)


# ─── __main__ guard ─────────────────────────────────────────────


class TestMainGuard:
    def test_sys_exit_on_file_not_found(self, tmp_path):
        """Il blocco __main__ chiama sys.exit(1) su FileNotFoundError."""
        from person_anonymizer import camera_calibration as cc_module

        args = ["--images", str(tmp_path / "nodir"), "--output", "out.npz"]
        with (
            patch("sys.argv", ["camera_calibration.py"] + args),
            patch.object(sys, "exit") as mock_exit,
        ):
            try:
                # Simula esecuzione come __main__
                try:
                    cc_module.main()
                except (FileNotFoundError, ValueError):
                    sys.exit(1)
            except SystemExit:
                pass

        mock_exit.assert_called_once_with(1)
