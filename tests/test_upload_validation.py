"""Tests for web.upload_validation."""

import json

import pytest

from person_anonymizer.web.upload_validation import (
    has_valid_video_signature,
    job_file_path,
    parse_annotations_json,
)

MAX_BYTES = 100 * 1024 * 1024


class TestHasValidVideoSignature:
    def test_mp4_header_with_mp4_extension_is_valid(self):
        assert has_valid_video_signature(b"\x00\x00\x00\x18ftypisom", ".mp4") is True

    def test_riff_header_with_avi_extension_is_valid(self):
        assert has_valid_video_signature(b"RIFF\x00\x00\x00\x00AVI ", ".avi") is True

    def test_mkv_header_with_mp4_extension_is_rejected(self):
        assert has_valid_video_signature(b"\x1a\x45\xdf\xa3\x00\x00", ".mp4") is False

    def test_mkv_header_with_mkv_extension_is_valid(self):
        assert has_valid_video_signature(b"\x1a\x45\xdf\xa3\x00\x00", ".mkv") is True

    def test_empty_header_is_rejected(self):
        assert has_valid_video_signature(b"", ".mp4") is False


class TestParseAnnotationsJson:
    def test_object_is_returned(self):
        content = json.dumps({"frames": {"0": []}}).encode()

        assert parse_annotations_json(content, max_bytes=MAX_BYTES) == {"frames": {"0": []}}

    def test_content_over_limit_raises_size_error(self):
        content = b" " * (MAX_BYTES + 1)

        with pytest.raises(ValueError, match=r"troppo grande \(max 100 MB\)"):
            parse_annotations_json(content, max_bytes=MAX_BYTES)

    def test_content_at_limit_is_not_a_size_error(self):
        content = b"{}" + b" " * (MAX_BYTES - 2)

        assert parse_annotations_json(content, max_bytes=MAX_BYTES) == {}

    def test_list_root_raises_object_error(self):
        with pytest.raises(ValueError, match="deve essere un oggetto"):
            parse_annotations_json(b"[]", max_bytes=MAX_BYTES)

    def test_non_utf8_bytes_raise_invalid_json_error(self):
        with pytest.raises(ValueError, match="non contiene JSON valido"):
            parse_annotations_json(b"\xff\xfe\xfa", max_bytes=MAX_BYTES)

    def test_malformed_json_raises_invalid_json_error(self):
        with pytest.raises(ValueError, match="non contiene JSON valido"):
            parse_annotations_json(b"{not json", max_bytes=MAX_BYTES)

    def test_integer_beyond_python_digit_limit_raises_invalid_json_error(self):
        # json.loads raises a plain ValueError (not JSONDecodeError) here; the
        # user must still get the Italian message, not Python's English one.
        content = b'{"a": ' + b"9" * 5000 + b"}"
        with pytest.raises(ValueError, match="non contiene JSON valido"):
            parse_annotations_json(content, max_bytes=MAX_BYTES)


class TestJobFilePath:
    def test_file_inside_base_returns_resolved_path(self, tmp_path):
        assert job_file_path(tmp_path, "job1", "v.mp4") == (tmp_path / "job1" / "v.mp4").resolve()

    def test_dotdot_escaping_base_returns_none(self, tmp_path):
        base = tmp_path / "uploads"
        base.mkdir()

        assert job_file_path(base, "job1", "../../x.mp4") is None

    def test_sibling_dir_with_common_prefix_is_rejected(self, tmp_path):
        base = tmp_path / "uploads"
        base.mkdir()
        (tmp_path / "uploads_evil").mkdir()

        assert job_file_path(base, "../uploads_evil", "x.mp4") is None

    def test_symlink_pointing_outside_base_is_rejected(self, tmp_path):
        base = tmp_path / "uploads"
        (base / "job1").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (base / "job1" / "link.mp4").symlink_to(outside / "secret.mp4")

        assert job_file_path(base, "job1", "link.mp4") is None
