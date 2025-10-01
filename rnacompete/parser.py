"""
parser.py

Argument parser for running the RNAcompete program.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """
    Build and configure the command-line argument parser for RNAcompete.

    Returns
    -------
    parser : argparse.ArgumentParser
        Configured parser for the RNAcompete program.
    """

    # ------------------------------------------------------------------
    # Main parser
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog='rnacompete',
        description='Process and summarize RNAcompete data.',
    )
    subparsers = parser.add_subparsers(
        title='mode',
        dest='mode',
        required=True,
        help='Available modes.',
    )

    # ------------------------------------------------------------------
    # Process mode
    # ------------------------------------------------------------------
    process_parser = subparsers.add_parser(
        'process',
        help='Process RNAcompete data for a single batch.',
    )
    process_parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Root directory containing batch folders.',
    )
    process_parser.add_argument(
        '--batch',
        type=str,
        required=True,
        help='Name of the batch folder to process.',
    )
    process_parser.add_argument(
        '--trim-fraction',
        dest='trim_fraction',
        type=float,
        default=0.025,
        help='Fraction of probe Z-scores to discard from both ends when '
             'computing trimmed means (default: 0.025).',
    )
    process_parser.add_argument(
        '--k',
        type=int,
        default=7,
        help='Length of the k-mers (default: 7).',
    )
    process_parser.add_argument(
        '--top-k',
        dest='top_k',
        type=int,
        default=10,
        help='Number of top k-mers used to generate the motif (default: 10).',
    )
    process_parser.add_argument(
        '--n-jobs',
        dest='n_jobs',
        type=int,
        default=1,
        help='Number of parallel jobs to run (default: 1).'
    )

    # ------------------------------------------------------------------
    # Summarize mode
    # ------------------------------------------------------------------
    summarize_parser = subparsers.add_parser(
        'summarize',
        help='Summarize RNAcompete results from multiple experiments.',
    )
    summarize_parser.add_argument(
        '--root',
        type=str,
        required=True,
        help='Root directory containing batch folders.',
    )
    summarize_parser.add_argument(
        '--output',
        dest='output_path',
        type=str,
        required=True,
        help='Directory to save summarized results and optional HTML reports.'
    )
    summarize_parser.add_argument(
        '--no-html',
        dest='no_html',
        action='store_true',
        help='Combine and save summary tables. only'
    )

    # Mutually exclusive options for summarize mode
    group = summarize_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--summarize-all',
        dest='summarize_all',
        action='store_true',
        help='Summarize all experiments across all batches in the root '
             'directory.'
    )
    group.add_argument(
        '--summarize-batch',
        dest='summarize_batch',
        type=str,
        help='Summarize all experiments within a single batch folder (provide '
             'batch name).',
    )
    group.add_argument(
        '--summarize-experiments',
        dest='summarize_experiments',
        type=str,
        help='Summarize selected experiments across all batches in the root '
             'directory (provide path to a file containing experiment IDs, '
             'one per line).'
    )

    return parser
