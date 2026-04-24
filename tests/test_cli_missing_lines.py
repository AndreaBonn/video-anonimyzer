"""
Test supplementari per cli.py — riga 105 (__main__ block).

La riga mancante è:
    if __name__ == "__main__":
        main()

Coprire questa riga richiede eseguire il modulo come script principale.
"""

import subprocess
import sys

import pytest


class TestCliMainBlock:
    """Verifica che il blocco __main__ chiami main()."""

    def test_module_execution_exits_1_for_missing_file(self):
        """python -m person_anonymizer.cli con file inesistente → exit 1."""
        result = subprocess.run(
            [sys.executable, "-m", "person_anonymizer.cli", "nonexistent.mp4"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Assert — exit 1 per file mancante, messaggio contiene il filename
        assert result.returncode == 1
        combined_output = result.stdout + result.stderr
        assert "nonexistent" in combined_output

    def test_no_args_shows_usage_and_exits_2(self):
        """Eseguire senza argomenti mostra usage/help e esce con codice 2."""
        result = subprocess.run(
            [sys.executable, "-m", "person_anonymizer.cli"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Assert — argparse esce con codice 2 e mostra usage
        assert result.returncode == 2
        assert "usage" in result.stderr.lower()
