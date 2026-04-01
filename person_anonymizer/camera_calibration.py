#!/usr/bin/env python3
"""
Utility di calibrazione camera per correzione distorsione fish-eye.

Usa immagini di una scacchiera (chessboard) per calcolare
i parametri intrinseci della camera.

Utilizzo:
    python camera_calibration.py --images ./calibration_images/ --output calibration.npz

Le immagini devono contenere una scacchiera stampata, fotografata
da diverse angolazioni con la stessa camera usata per la sorveglianza.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

__all__ = ["find_chessboard_corners", "calibrate_camera"]

_log = logging.getLogger(__name__)


def find_chessboard_corners(image_paths, board_size=(9, 6)):
    """
    Trova gli angoli della scacchiera in una serie di immagini.

    Parameters
    ----------
    image_paths : list of str
        Percorsi delle immagini di calibrazione.
    board_size : tuple of int
        Dimensione interna della scacchiera (colonne, righe).

    Returns
    -------
    tuple (obj_points, img_points, image_size)
    """
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)

    obj_points = []
    img_points = []
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            _log.warning("Impossibile leggere %s, saltato", path)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners)
            _log.info("  OK: %s", Path(path).name)
        else:
            _log.info("  SKIP: scacchiera non trovata in %s", Path(path).name)

    return obj_points, img_points, image_size


def calibrate_camera(obj_points, img_points, image_size):
    """
    Calibra la camera e restituisce matrice e coefficienti di distorsione.

    Returns
    -------
    tuple (camera_matrix, dist_coefficients, rms_error)
    """
    ret, camera_matrix, dist_coefficients, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return camera_matrix, dist_coefficients, ret


def main():
    parser = argparse.ArgumentParser(description="Calibrazione camera per correzione fish-eye")
    parser.add_argument("--images", required=True, help="Cartella con immagini di calibrazione")
    parser.add_argument("--output", default="calibration.npz", help="File di output (.npz)")
    parser.add_argument(
        "--board-cols", type=int, default=9, help="Colonne interne scacchiera (default: 9)"
    )
    parser.add_argument(
        "--board-rows", type=int, default=6, help="Righe interne scacchiera (default: 6)"
    )
    args = parser.parse_args()

    if not Path(args.images).is_dir():
        raise FileNotFoundError(f"Cartella non trovata: {args.images}")

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(Path(args.images).glob(ext)))

    if not image_paths:
        raise FileNotFoundError(f"Nessuna immagine trovata in {args.images}")

    image_paths.sort()
    print("\nCalibrazione camera")
    print(f"  Immagini trovate: {len(image_paths)}")
    print(f"  Scacchiera: {args.board_cols}x{args.board_rows}")
    print()

    board_size = (args.board_cols, args.board_rows)
    obj_points, img_points, image_size = find_chessboard_corners(image_paths, board_size)

    if len(obj_points) < 3:
        raise ValueError(f"Servono almeno 3 immagini valide, trovate solo {len(obj_points)}")

    print(f"\nImmagini valide: {len(obj_points)} / {len(image_paths)}")
    print("Calibrazione in corso...")

    camera_matrix, dist_coefficients, rms_error = calibrate_camera(
        obj_points, img_points, image_size
    )

    np.savez(args.output, camera_matrix=camera_matrix, dist_coefficients=dist_coefficients)

    print(f"\nCalibrazione completata!")
    print(f"  Errore RMS: {rms_error:.4f}")
    print(f"  File salvato: {args.output}")
    print(f"\nPer usare i parametri in person_anonymizer.py:")
    print(f"  CAMERA_MATRIX = np.{repr(camera_matrix)}")
    print(f"  DIST_COEFFICIENTS = np.{repr(dist_coefficients.ravel())}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nErrore: {e}")
        sys.exit(1)
