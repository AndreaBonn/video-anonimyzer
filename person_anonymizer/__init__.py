"""Person Anonymizer — Tool di anonimizzazione persone in video di sorveglianza."""

from .config import VERSION
from .models import PipelineError, PipelineInputError

__version__ = VERSION
__all__ = ["VERSION", "__version__", "PipelineError", "PipelineInputError"]
