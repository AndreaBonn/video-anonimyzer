"""CLI entry point per Person Anonymizer."""

import argparse
import sys

from .config import VERSION
from .models import PipelineContext, PipelineError, PipelineInputError

__all__ = ["parse_args", "main"]


def parse_args():
    """Parser argomenti CLI.

    Returns
    -------
    argparse.Namespace
        Argomenti parsati dalla riga di comando.
    """
    parser = argparse.ArgumentParser(
        description=(
            f"Person Anonymizer v{VERSION} — "
            "Oscuramento automatico persone in video di sorveglianza"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Percorso del video da elaborare")
    parser.add_argument(
        "-M",
        "--mode",
        choices=["manual", "auto"],
        default=None,
        help="Modalità operativa (default: da config)",
    )
    parser.add_argument("-o", "--output", default=None, help="Percorso file di output")
    parser.add_argument(
        "-m",
        "--method",
        choices=["pixelation", "blur"],
        default=None,
        help="Metodo di oscuramento (default: da config)",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disabilita video debug")
    parser.add_argument("--no-report", action="store_true", help="Disabilita CSV report")
    parser.add_argument(
        "--review",
        default=None,
        help=(
            "Ricarica annotazioni da JSON esistente, "
            "salta la detection e apre solo la revisione manuale"
        ),
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Normalizza i poligoni in rettangoli e unifica "
            "le aree sovrapposte. Richiede --review."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["yolo", "yolo+sam3", "sam3"],
        default="yolo",
        help=(
            "Backend di detection: 'yolo' (default), "
            "'yolo+sam3' (YOLO + SAM3 refiner), "
            "'sam3' (SAM3 completo). SAM3 richiede installazione separata."
        ),
    )
    return parser.parse_args()


def main():
    """Entry point CLI."""
    args = parse_args()
    ctx = PipelineContext(
        input=args.input,
        mode=args.mode,
        method=args.method,
        output=args.output,
        no_debug=args.no_debug,
        no_report=args.no_report,
        review=args.review,
        normalize=args.normalize,
    )
    try:
        from .config import PipelineConfig
        from .pipeline import run_pipeline

        config = PipelineConfig(detection_backend=args.backend)
        run_pipeline(ctx, config=config)
    except PipelineInputError as e:
        print(f"\nErrore: {e}")
        sys.exit(1)
    except PipelineError as e:
        print(f"\nErrore pipeline: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrotto dall'utente (Ctrl+C).")
        sys.exit(1)


if __name__ == "__main__":
    main()
