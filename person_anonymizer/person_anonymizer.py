"""Facade per backward compatibility — reindirizza al modulo pipeline.

Questo file è mantenuto per compatibilità con codice che importa direttamente
da person_anonymizer (es. web/pipeline_runner.py, tests).
"""

from .models import PipelineError, PipelineInputError  # noqa: F401
from .pipeline import run_pipeline  # noqa: F401

__all__ = ["run_pipeline", "PipelineError", "PipelineInputError"]
