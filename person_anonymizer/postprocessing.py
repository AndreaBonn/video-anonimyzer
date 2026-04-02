"""
Post-processing del video anonimizzato.

Contiene le funzioni per la codifica H.264 con/senza reintegro audio,
la verifica post-rendering tramite secondo passaggio YOLO e il filtro
degli artefatti di pixelazione.

La normalizzazione delle annotazioni è in normalization.py.
"""

import logging
import shutil

import cv2
import ffmpeg
from tqdm import tqdm

from .anonymization import polygon_to_bbox
from .config import PipelineConfig
from .detection import apply_nms, compute_iou_boxes, detect_and_rescale
from .normalization import normalize_annotations  # noqa: F401

__all__ = [
    "encode_with_audio",
    "encode_without_audio",
    "run_post_render_check",
    "filter_artifact_detections",
    "normalize_annotations",
]

_log = logging.getLogger(__name__)


def encode_with_audio(video_no_audio, original_video, output_path):
    """
    Codifica il video in H.264 (libx264) con qualità alta e reintegra l'audio.

    Usa CRF 18 (visually lossless) e preset 'slow' per massima qualità.
    Il codec mp4v usato da cv2.VideoWriter è MPEG-4 Part 2 (2001) e degrada
    significativamente la qualità; H.264 offre qualità nettamente superiore
    a parità di bitrate.
    """
    try:
        video_in = ffmpeg.input(video_no_audio)
        audio_in = ffmpeg.input(original_video).audio
        (
            ffmpeg.output(
                video_in,
                audio_in,
                output_path,
                vcodec="libx264",
                crf=18,
                preset="slow",
                pix_fmt="yuv420p",
                acodec="aac",
                audio_bitrate="192k",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        _log.warning("ffmpeg con audio fallito, tentativo senza audio: %s", e)
        # Fallback: tenta senza audio
        try:
            video_in = ffmpeg.input(video_no_audio)
            (
                ffmpeg.output(
                    video_in,
                    output_path,
                    vcodec="libx264",
                    crf=18,
                    preset="slow",
                    pix_fmt="yuv420p",
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            _log.warning("ffmpeg completamente fallito, copia grezza AVI: %s", e)
            shutil.copy(video_no_audio, output_path)


def encode_without_audio(video_no_audio, output_path):
    """Codifica il video in H.264 senza audio."""
    try:
        video_in = ffmpeg.input(video_no_audio)
        (
            ffmpeg.output(
                video_in, output_path, vcodec="libx264", crf=18, preset="slow", pix_fmt="yuv420p"
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        _log.warning("ffmpeg fallito per debug video, copia grezza AVI: %s", e)
        shutil.copy(video_no_audio, output_path)


def run_post_render_check(
    anonymized_video_path, model, confidence, report_data, config: PipelineConfig, check_scales=None
):
    """
    Secondo passaggio YOLO sul video oscurato, con multi-scale.

    Usa le stesse scale della detection originale per intercettare
    anche persone piccole che una singola scala non rileva.

    Parameters
    ----------
    anonymized_video_path : str
        Percorso video anonimizzato.
    model : YOLO
        Modello YOLO caricato.
    confidence : float
        Soglia di confidenza.
    report_data : dict
        Dati report per aggiornamento alerting.
    config : PipelineConfig
        Configurazione della pipeline (soglia NMS).
    check_scales : list of float, optional
        Scale di verifica (default: [1.0, 2.0]).

    Returns
    -------
    list of tuple
        Ogni tupla: (frame_idx, count, nms_boxes) dove nms_boxes è la lista
        dei box [x1, y1, x2, y2, conf] rilevati nel frame.
    """
    if check_scales is None:
        check_scales = [1.0, 2.0]

    cap = cv2.VideoCapture(anonymized_video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    alert_frames = []
    frame_idx = 0

    try:
        pbar = tqdm(total=total, desc="Verifica", unit=" frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            all_boxes = []
            for scale in check_scales:
                if scale == 1.0:
                    results = model(frame, conf=confidence, classes=[0], verbose=False)
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        all_boxes.append([x1, y1, x2, y2, float(box.conf[0])])
                else:
                    new_w = int(frame_w * scale)
                    new_h = int(frame_h * scale)
                    scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    boxes = detect_and_rescale(model, scaled, confidence, scale)
                    all_boxes.extend(boxes)

            nms_boxes = apply_nms(all_boxes, config.nms_iou_threshold)

            if len(nms_boxes) > 0:
                alert_frames.append((frame_idx, len(nms_boxes), nms_boxes))
                if frame_idx in report_data:
                    report_data[frame_idx]["post_check_alerts"] = len(nms_boxes)

            frame_idx += 1
            pbar.update(1)
        pbar.close()
    finally:
        cap.release()
    return alert_frames


def filter_artifact_detections(alert_frames, annotations, iou_threshold):
    """
    Filtra rilevamenti post-render che sono artefatti della pixelazione.

    Un rilevamento che si sovrappone (IoU >= soglia) con un'area già
    anonimizzata è quasi certamente un artefatto: YOLO interpreta la
    pixelazione come una persona. I rilevamenti genuini sono quelli che
    NON si sovrappongono significativamente con aree già oscurate.

    Parameters
    ----------
    alert_frames : list of tuple
        Ogni tupla: (frame_idx, count, nms_boxes).
    annotations : dict
        Annotazioni correnti {frame_idx: {"auto": [...], "manual": [...]}}.
    iou_threshold : float
        Soglia IoU per considerare un rilevamento come artefatto.

    Returns
    -------
    genuine_alerts : list of tuple
        (frame_idx, genuine_boxes) solo frame con rilevamenti genuini.
    total_artifacts : int
        Numero totale di artefatti filtrati.
    total_genuine : int
        Numero totale di rilevamenti genuini.
    """
    genuine_alerts = []
    total_artifacts = 0
    total_genuine = 0

    for frame_idx, count, nms_boxes in alert_frames:
        ann = annotations.get(frame_idx, {"auto": [], "manual": []})
        existing_bboxes = []
        for poly in ann.get("auto", []):
            existing_bboxes.append(polygon_to_bbox(poly))
        for poly in ann.get("manual", []):
            existing_bboxes.append(polygon_to_bbox(poly))

        genuine_boxes = []
        for det_box in nms_boxes:
            det_bbox = det_box[:4]
            is_artifact = False
            for ex_bbox in existing_bboxes:
                if compute_iou_boxes(det_bbox, ex_bbox) >= iou_threshold:
                    is_artifact = True
                    break
            if is_artifact:
                total_artifacts += 1
            else:
                genuine_boxes.append(det_box)
                total_genuine += 1

        if genuine_boxes:
            genuine_alerts.append((frame_idx, genuine_boxes))

    return genuine_alerts, total_artifacts, total_genuine
