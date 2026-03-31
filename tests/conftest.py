"""Configurazione pytest — aggiunge person_anonymizer al path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "person_anonymizer"))
