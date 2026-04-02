"""Cattura stdout e tqdm per inoltro via SSE."""

import re
import sys
import time

from person_anonymizer.web.sse_manager import SSEManager

__all__ = ["TqdmCapture", "StdoutCapture"]


class TqdmCapture:
    """Monkey-patch tqdm per catturare il progresso ed emetterlo via SSE."""

    def __init__(self, sse: SSEManager, job_id: str):
        self._sse = sse
        self._job_id = job_id
        self._original_tqdm = None

    def install(self):
        """Installa il patch su tqdm (nel modulo tqdm e in pipeline_stages)."""
        import tqdm as tqdm_module

        import person_anonymizer.pipeline_stages as pa_stages

        self._original_tqdm = tqdm_module.tqdm

        sse = self._sse
        job_id = self._job_id

        class PatchedTqdm(self._original_tqdm):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._sse = sse
                self._job_id = job_id
                self._last_emit = 0
                # Emetti evento inizio fase
                desc = self.desc or ""
                sse.emit(
                    job_id,
                    "phase",
                    {
                        "description": desc,
                        "total": self.total or 0,
                    },
                )

            def update(self, n=1):
                super().update(n)
                now = time.monotonic()
                # Rate-limit: emetti max ogni 0.25s
                if now - self._last_emit >= 0.25:
                    self._last_emit = now
                    rate = self.format_dict.get("rate", 0) or 0
                    elapsed = self.format_dict.get("elapsed", 0) or 0
                    self._sse.emit(
                        self._job_id,
                        "progress",
                        {
                            "current": self.n,
                            "total": self.total or 0,
                            "description": self.desc or "",
                            "rate": round(rate, 2),
                            "elapsed": round(elapsed, 1),
                        },
                    )

            def close(self):
                # Emetti progresso finale
                self._sse.emit(
                    self._job_id,
                    "progress",
                    {
                        "current": self.total or self.n,
                        "total": self.total or self.n,
                        "description": self.desc or "",
                        "rate": 0,
                        "elapsed": 0,
                    },
                )
                super().close()

        # Patcha sia il modulo tqdm che il riferimento in pipeline_stages
        tqdm_module.tqdm = PatchedTqdm
        pa_stages.tqdm = PatchedTqdm

    def uninstall(self):
        """Ripristina tqdm originale."""
        if self._original_tqdm:
            import tqdm as tqdm_module

            import person_anonymizer.pipeline_stages as pa_stages

            tqdm_module.tqdm = self._original_tqdm
            pa_stages.tqdm = self._original_tqdm


class StdoutCapture:
    """Cattura stdout e invia le righe come eventi SSE 'log'."""

    _PATH_RE = re.compile(r"/[^\s]*/(uploads|outputs)/[^\s]+")
    _PHASE_RE = re.compile(r"\[FASE (\d)/5\]")

    def __init__(self, sse: SSEManager, job_id: str):
        self._sse = sse
        self._job_id = job_id
        self._original = None
        self._buffer = ""

    @classmethod
    def _sanitize_message(cls, msg: str) -> str:
        """Rimuove path assoluti dai messaggi di log inviati al client."""
        return cls._PATH_RE.sub("[FILE]", msg)

    def install(self):
        self._original = sys.stdout
        sys.stdout = self

    def uninstall(self):
        if self._original:
            # Emetti eventuale testo residuo nel buffer
            if self._buffer.strip():
                sanitized = self._sanitize_message(self._buffer.strip())
                self._sse.emit(self._job_id, "log", {"message": sanitized})
                self._buffer = ""
            sys.stdout = self._original

    def write(self, text):
        if self._original:
            self._original.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                sanitized = self._sanitize_message(line)
                # Detecta fasi dalla stampa
                phase_match = self._PHASE_RE.match(sanitized)
                if phase_match:
                    self._sse.emit(
                        self._job_id,
                        "phase_label",
                        {
                            "phase": int(phase_match.group(1)),
                            "label": sanitized,
                        },
                    )
                self._sse.emit(self._job_id, "log", {"message": sanitized})

    def flush(self):
        if self._original:
            self._original.flush()
