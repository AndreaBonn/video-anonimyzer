"""Per-frame report rows and cumulative statistics of the detection loop."""

from dataclasses import dataclass, field

import numpy as np

__all__ = ["DetectionTotals", "frame_confidences", "report_row"]

# The summary lists at most this many corrupted frame indices
_MAX_LISTED_CORRUPTED = 10


def frame_confidences(result):
    """Confidenze dei box tracciati di un frame."""
    return [conf for *_, conf in result.tracked]


@dataclass
class DetectionTotals:
    """Contatori cumulativi del loop, aggiornati frame dopo frame."""

    unique_ids: set = field(default_factory=set)
    total_instances: int = 0
    frames_zero_det: int = 0
    all_confs: list = field(default_factory=list)
    corrupted: list = field(default_factory=list)

    def record(self, result):
        """Accumula ID, confidenze e conteggi di un frame elaborato."""
        self.unique_ids.update(tid for tid, *_ in result.tracked)
        self.all_confs.extend(frame_confidences(result))
        self.total_instances += len(result.polygons)
        if not result.polygons:
            self.frames_zero_det += 1

    def summary_lines(self, total_frames):
        """Righe del riepilogo finale, senza la riga vuota iniziale.

        Parameters
        ----------
        total_frames : int
            Numero totale di frame del video.

        Returns
        -------
        list[str]
            Righe da stampare nell'ordine.
        """
        avg_conf = float(np.mean(self.all_confs)) if self.all_confs else 0.0
        zero_pct = (self.frames_zero_det / total_frames * 100) if total_frames > 0 else 0
        lines = [
            f"  Persone tracciate (ID unici):   {len(self.unique_ids)}",
            f"  Istanze totali rilevate:        {self.total_instances:,}",
            f"  Frame con 0 rilevamenti:        {self.frames_zero_det}  ({zero_pct:.1f}%)",
            f"  Confidenza media:               {avg_conf:.2f}",
        ]
        if self.corrupted:
            more = "..." if len(self.corrupted) > _MAX_LISTED_CORRUPTED else ""
            lines.append(
                f"  ATTENZIONE: {len(self.corrupted)} frame corrotti (saltati): "
                f"{self.corrupted[:_MAX_LISTED_CORRUPTED]}{more}"
            )
        return lines


def report_row(frame_idx, result):
    """Riga di report per un frame; confidenze assenti danno 0.0."""
    confs = frame_confidences(result)
    return {
        "frame_number": frame_idx,
        "persons_detected": len(result.polygons),
        "avg_confidence": float(np.mean(confs)) if confs else 0.0,
        "min_confidence": float(min(confs)) if confs else 0.0,
        "max_confidence": float(max(confs)) if confs else 0.0,
        "motion_zones": result.motion_count,
        "sliding_window_hits": result.sw_hits,
        "multiscale_hits": result.ms_hits,
        "post_check_alerts": 0,
    }
