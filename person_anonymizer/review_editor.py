"""Pure editing core of the manual review: annotations, navigation and polygon drawing state."""

from copy import deepcopy

import cv2
import numpy as np

__all__ = ["ReviewEditor", "new_annotation", "point_in_polygon"]

MIN_POLYGON_POINTS = 3

Point = tuple[int, int]


def new_annotation() -> dict:
    """Return an empty per-frame annotation entry."""
    return {"auto": [], "manual": [], "intensities": []}


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """
    Check whether a point lies inside or on the border of a polygon.

    Parameters
    ----------
    point : tuple[int, int]
        Point (x, y) to test.
    polygon : list[tuple[int, int]]
        Polygon vertices.

    Returns
    -------
    bool
        True if the point is inside or on the border.
    """
    pts = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(pts, point, False) >= 0


class ReviewEditor:
    """
    State and rules of a manual review session, with no I/O.

    Annotations, current frame, pending polygon points, delete mode and
    stats live together because every action changes several of them at once
    (navigating clears the pending polygon and delete mode, closing a polygon
    updates the stats).
    """

    def __init__(self, annotations: dict, total_frames: int) -> None:
        self.annotations = deepcopy(annotations)
        self.total_frames = total_frames
        self.current_frame_idx = 0
        self.pending_points: list[Point] = []
        self.delete_mode = False
        self.stats = {
            "added": 0,
            "removed": 0,
            "frames_modified": set(),
            "frames_reviewed": set(),
        }

    def visit_current(self) -> dict:
        """Mark the current frame as reviewed and return its annotation, creating it if missing."""
        self.stats["frames_reviewed"].add(self.current_frame_idx)
        return self.annotations.setdefault(self.current_frame_idx, new_annotation())

    def current_annotation(self) -> dict:
        """Return the current frame annotation, or an unstored empty one if the frame has none."""
        return self.annotations.get(self.current_frame_idx, new_annotation())

    def click(self, point: Point) -> None:
        """Delete the polygon under the point in delete mode, otherwise add a polygon point."""
        if self.delete_mode:
            self.delete_at(point)
        else:
            self.add_point(point)

    def add_point(self, point: Point) -> None:
        """Append a point to the polygon being drawn."""
        self.pending_points.append(point)

    def close_polygon(self) -> bool:
        """
        Store the pending points as a manual polygon on the current frame.

        Returns
        -------
        bool
            False, leaving the pending points untouched, if there are fewer
            than three of them.
        """
        if len(self.pending_points) < MIN_POLYGON_POINTS:
            return False
        annotation = self.annotations.setdefault(self.current_frame_idx, new_annotation())
        annotation["manual"].append(list(self.pending_points))
        self.stats["added"] += 1
        self.stats["frames_modified"].add(self.current_frame_idx)
        self.pending_points = []
        return True

    def undo_point(self) -> None:
        """Drop the last pending point, if any."""
        if self.pending_points:
            self.pending_points.pop()

    def cancel_polygon(self) -> None:
        """Discard the polygon being drawn."""
        self.pending_points = []

    def toggle_delete_mode(self) -> None:
        """Flip delete mode and discard the polygon being drawn."""
        self.delete_mode = not self.delete_mode
        self.pending_points = []

    def delete_at(self, point: Point) -> bool:
        """
        Remove the first polygon containing the point, manual ones before auto ones.

        Removing an auto polygon also removes its intensity when one is aligned with it.

        Returns
        -------
        bool
            True if a polygon was removed.
        """
        annotation = self.annotations.get(self.current_frame_idx)
        if annotation is None:
            return False
        for i, polygon in enumerate(annotation.get("manual", [])):
            if point_in_polygon(point, polygon):
                annotation["manual"].pop(i)
                self._record_removal()
                return True
        for i, polygon in enumerate(annotation.get("auto", [])):
            if point_in_polygon(point, polygon):
                annotation["auto"].pop(i)
                intensities = annotation.get("intensities", [])
                if i < len(intensities):
                    intensities.pop(i)
                self._record_removal()
                return True
        return False

    def go_next(self) -> None:
        """Move to the next frame, staying on the last one, and reset the interaction state."""
        self._reset_interaction()
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1

    def go_previous(self) -> None:
        """Move to the previous frame, staying on the first one, and reset the interaction state."""
        self._reset_interaction()
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1

    def final_stats(self) -> dict:
        """Return the session stats with the frame sets reduced to counts."""
        return {
            "added": self.stats["added"],
            "removed": self.stats["removed"],
            "frames_modified": len(self.stats["frames_modified"]),
            "frames_reviewed": len(self.stats["frames_reviewed"]),
        }

    def _record_removal(self) -> None:
        self.stats["removed"] += 1
        self.stats["frames_modified"].add(self.current_frame_idx)

    def _reset_interaction(self) -> None:
        self.pending_points = []
        self.delete_mode = False
