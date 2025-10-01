"""
cli.py

Command-line interface for processing and summarizing RNAcompete data.
"""

import logging
import sys

from . import run_process, run_summarize
from .parser import build_parser


def main_cli() -> None:
    """
    Main entry point for the RNAcompete CLI.

    Parses command-line arguments and dispatches execution to the appropriate
    core function based on the selected subcommand ('process' or 'summarize').
    """

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Parse command-line arguments
    parser = build_parser()
    args = parser.parse_args()

    # Process mode
    if args.mode == 'process':

        # Call the core API
        run_process(
            root=args.root,
            batch=args.batch,
            trim_fraction=args.trim_fraction,
            k=args.k,
            top_k=args.top_k,
            n_jobs=args.n_jobs,
            save_results=True,
            return_results=False
        )

    # Summarize mode
    elif args.mode == 'summarize':

        # Call the core API
        run_summarize(
            root=args.root,
            output_path=args.output_path,
            no_html=args.no_html,
            summarize_all=args.summarize_all,
            summarize_batch=args.summarize_batch,
            summarize_experiments=args.summarize_experiments,
            save_results=True,
            return_results=False
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main_cli()
