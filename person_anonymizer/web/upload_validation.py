"""Pure validation helpers for uploaded files and job paths."""

import json
from pathlib import Path

_BYTES_PER_MB = 1024 * 1024

# Magic-byte signatures of the supported video containers
_VIDEO_SIGNATURES: dict[bytes, set[str]] = {
    b"\x00\x00\x00": {".mp4", ".m4v", ".mov"},  # ftyp container (ISO BMFF)
    b"\x1a\x45\xdf": {".mkv", ".webm"},  # EBML header (Matroska)
    b"RIFF": {".avi"},  # RIFF container
}


def has_valid_video_signature(header: bytes, ext: str) -> bool:
    """Check that the file header matches the container expected for ``ext``.

    Parameters
    ----------
    header : bytes
        First bytes of the uploaded file.
    ext : str
        Lowercase extension including the dot (e.g. ``".mp4"``).

    Returns
    -------
    bool
        True if a known signature prefixes ``header`` and allows ``ext``.
    """
    return any(
        header.startswith(signature) and ext in valid_exts
        for signature, valid_exts in _VIDEO_SIGNATURES.items()
    )


def parse_annotations_json(content: bytes, max_bytes: int) -> dict:
    """Parse an annotations upload, rejecting oversized or non-object JSON.

    Parameters
    ----------
    content : bytes
        Raw uploaded bytes.
    max_bytes : int
        Maximum accepted size in bytes.

    Returns
    -------
    dict
        The parsed JSON object.

    Raises
    ------
    ValueError
        With the user-facing message when the content is too large,
        not valid JSON/UTF-8, or not a JSON object.
    """
    if len(content) > max_bytes:
        raise ValueError(f"File JSON troppo grande (max {max_bytes // _BYTES_PER_MB} MB)")
    try:
        parsed = json.loads(content)
    # JSONDecodeError, UnicodeDecodeError and the int digit limit are all ValueError
    except ValueError:
        raise ValueError("Il file non contiene JSON valido") from None
    if not isinstance(parsed, dict):
        raise ValueError("Il JSON deve essere un oggetto")
    return parsed


def job_file_path(base: Path, job_id: str, safe_name: str) -> Path | None:
    """Resolve ``base/job_id/safe_name`` and confirm it stays inside ``base``.

    Returns
    -------
    Path | None
        The resolved path, or None when it escapes ``base``.
    """
    resolved = (base / job_id / safe_name).resolve()
    if resolved.is_relative_to(base.resolve()):
        return resolved
    return None
