"""Fase di auto-refinement: loop di ri-rendering e verifica post-render."""

from .anonymization import box_to_polygon, resolve_intensity
from .postprocessing import filter_artifact_detections, run_post_render_check
from .rendering import render_video

__all__ = ["run_refinement_loop"]


def run_refinement_loop(
    input_path,
    annotations,
    model,
    config,
    fps,
    frame_w,
    frame_h,
    method,
    fisheye,
    report_data,
    temp_video_path,
    stop_event=None,
):
    """Loop di auto-refinement.

    Parameters
    ----------
    input_path : str
        Percorso del video sorgente.
    annotations : dict
        Annotazioni correnti.
    model : YOLO
        Modello di detection.
    config : PipelineConfig
        Configurazione della pipeline.
    fps : float
        Frame rate del video.
    frame_w : int
        Larghezza frame.
    frame_h : int
        Altezza frame.
    method : str
        Metodo di anonimizzazione.
    fisheye : FisheyeContext
        Contesto di correzione fish-eye.
    report_data : dict
        Dati del report frame-per-frame.
    temp_video_path : str
        Percorso del video temporaneo.
    stop_event : threading.Event | None
        Evento di stop.

    Returns
    -------
    tuple
        (annotations, actual_passes, annotations_added)
    """
    if not config.enable_post_render_check:
        print(f"\n[FASE 2/5] Auto refinement — saltato (verifica disabilitata)")
        return annotations, 0, 0

    actual_passes, annotations_added = 0, 0
    for pass_num in range(1, config.max_refinement_passes + 1):
        if stop_event is not None and stop_event.is_set():
            print("\n  Pipeline interrotta dall'utente.")
            break

        actual_passes = pass_num
        pass_label = f"pass {pass_num}/{config.max_refinement_passes}"

        print(f"\n[FASE 2/5] Auto refinement — rendering ({pass_label})...")
        render_video(
            input_path,
            temp_video_path,
            annotations,
            fps,
            frame_w,
            frame_h,
            method,
            fisheye,
            config,
            debug_path=None,
            desc=f"Rendering ({pass_label})",
            stop_event=stop_event,
        )

        print(f"\n  Verifica post-rendering ({pass_label})...")
        alert_frames = run_post_render_check(
            temp_video_path, model, config.post_render_check_confidence, report_data, config
        )

        if not alert_frames:
            print(f"\n  Nessun rilevamento residuo — rendering OK.")
            break

        genuine_alerts, n_artifacts, n_genuine = filter_artifact_detections(
            alert_frames, annotations, config.refinement_overlap_threshold
        )
        print(f"\n  Rilevamenti post-render: {n_artifacts + n_genuine}")
        print(f"  Artefatti filtrati (IoU >= {config.refinement_overlap_threshold}): {n_artifacts}")
        print(f"  Residui genuini: {n_genuine}")

        if not genuine_alerts:
            print(f"  Tutti i rilevamenti sono artefatti della pixelazione — rendering OK.")
            break

        if pass_num == config.max_refinement_passes:
            print(f"\n  Raggiunto limite di {config.max_refinement_passes} pass.")
            print(f"  Residui genuini rimasti in {len(genuine_alerts)} frame:")
            for fidx, boxes in genuine_alerts:
                print(f"    Frame {fidx}: {len(boxes)} persona/e")
            print(
                "  -> Verranno mostrati nella revisione manuale."
                if method == "manual"
                else "  -> Rieseguire con --mode manual per correzione."
            )
            break

        added_this_pass = 0
        for fidx, boxes in genuine_alerts:
            if fidx not in annotations:
                annotations[fidx] = {"auto": [], "manual": [], "intensities": []}
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                annotations[fidx]["auto"].append(
                    box_to_polygon(
                        x1,
                        y1,
                        x2,
                        y2,
                        padding=config.person_padding,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        config=config,
                    )
                )
                annotations[fidx]["intensities"].append(resolve_intensity(config, y2 - y1))
                added_this_pass += 1

        annotations_added += added_this_pass
        print(f"\n  Aggiunte {added_this_pass} annotazioni — ri-rendering in corso...")

    return annotations, actual_passes, annotations_added
