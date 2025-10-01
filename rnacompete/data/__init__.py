"""
rnacompete/data/__init__.py

Enables the loading of probe metadata.
"""

from importlib import resources
import pandas as pd


def load_probe_metadata() -> pd.DataFrame:
    """
    Return the probe_metadata DataFrame.

    Returns
    -------
    probe_metadata_df : pd.DataFrame
        A (num_probe, 2) dataframe containing probe sequences and their set
        assignments.

    """
    with resources.open_text(__package__, 'probe_metadata.tsv') as f:
        return pd.read_csv(f, sep="\t", index_col=0)
