"""
Test per le eccezioni custom della pipeline.

Verifica che PipelineError e PipelineInputError siano correttamente
definite e utilizzabili come eccezioni standard Python.
"""

import pytest

from person_anonymizer import PipelineError, PipelineInputError


class TestPipelineExceptions:
    """Verifica le eccezioni custom della pipeline."""

    def test_pipeline_error_is_exception(self):
        # Arrange / Act / Assert
        assert issubclass(PipelineError, Exception)

    def test_pipeline_input_error_is_pipeline_error(self):
        # Arrange / Act / Assert
        assert issubclass(PipelineInputError, PipelineError)

    def test_pipeline_input_error_carries_message(self):
        # Arrange / Act
        err = PipelineInputError("file non trovato")

        # Assert
        assert str(err) == "file non trovato"

    def test_pipeline_error_catchable_as_exception(self):
        # Arrange / Act / Assert
        with pytest.raises(Exception):
            raise PipelineError("test")

    def test_pipeline_input_error_catchable_as_pipeline_error(self):
        # Arrange / Act / Assert
        with pytest.raises(PipelineError):
            raise PipelineInputError("test")
