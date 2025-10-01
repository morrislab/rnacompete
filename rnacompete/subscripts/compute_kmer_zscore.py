"""
compute_kmer_zscore.py

Compute k-mer Z-scores from probe Z-scores for probes in set A, set B, and set
AB.

Each row corresponds to a probe/k-mer, and each column corresponds to an
experiment.

The computation proceeds as follows:
(1) Negative probe Z-scores are set to zeros.
(2) For each k-mer and experiment, compute the trimmed mean of the Z-scores
    from all probes containing that k-mer, discarding a specified fraction of
    the lowest and highest values.
(3) Normalize the resulting trimmed means across k-mers by centering
    (mean = 0) and scaling (standard deviation = 1).
"""

import itertools
from typing import List, Tuple

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, find, vstack


def compute_kmer_count(
        probe_metadata_df: pd.DataFrame, k: int, n_jobs: int
) -> Tuple[List[str], csc_matrix]:
    """Count k-mers in probe sequences.

    Parameters
    ----------
    probe_metadata_df : pd.DataFrame
        A (num_probe, 2) dataframe containing probe sequences and their set
        assignments.
    k : int
        Length of the k-mers.
    n_jobs : int
        Number of parallel jobs to run.

    Returns
    -------
    kmer_list : List[str]
        List of k-mers.
    kmer_count_mat : csc_matrix
        A (num_probe, num_kmer) sparse matrix containing k-mer counts per
        probe.
    """

    # Generate all possible k-mers
    nuc_list = ['A', 'C', 'G', 'U']
    kmer_list = [''.join(i) for i in itertools.product(nuc_list, repeat=k)]
    kmer_dict = {kmer: idx for idx, kmer in enumerate(kmer_list)}
    num_kmer = len(kmer_list)

    # Process the probes in parallel
    def compute_single_count(probe_seq: str) -> csc_matrix:
        row = np.zeros((1, num_kmer), dtype=np.uint8)
        for start_idx in range(len(probe_seq) - k + 1):
            kmer = probe_seq[start_idx:start_idx + k]
            row[0, kmer_dict[kmer]] += 1

        return csc_matrix(row)

    row_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_count)(seq) for seq in probe_metadata_df['seq']
    )
    kmer_count_mat = vstack(row_list, format='csc')

    return kmer_list, kmer_count_mat


def trimmed_mean(a: np.ndarray, proportiontocut: float) -> float:
    """
    Compute trimmed mean with MATLAB-style rounding.

    Parameters
    ----------
    a : np.ndarray
        Input array of values.
    proportiontocut : float
        Fraction of values to cut from each tail.

    Returns
    -------
    mean : float
        Trimmed mean of the input array
    """
    a = a[~np.isnan(a)]
    nobs = len(a)
    if nobs == 0:
        return np.nan
    k = proportiontocut * nobs
    lowercut = int(np.ceil(k - 0.5))
    uppercut = nobs - lowercut
    atmp = np.partition(a, (lowercut, uppercut - 1))

    return np.mean(atmp[lowercut:uppercut])


def compute_kmer_zscore_set(
        probe_zscore_mat: np.ndarray,
        kmer_count_mat: csc_matrix,
        trim_fraction: float,
        n_jobs: int
) -> np.ndarray:
    """Compute k-mer Z-scores from probe Z-scores for a single probe set.

    Parameters
    ----------
    probe_zscore_mat : np.ndarray
        A (num_probe_set, num_exp) matrix containing probe Z-scores.
    kmer_count_mat : csc_matrix
        A (num_probe, num_kmer) sparse matrix of k-mer counts per probe.
    trim_fraction : float
        Fraction of probe Z-scores to discard from both ends when computing
        trimmed means.
    n_jobs : int
        Number of parallel jobs to run.

    Returns
    -------
    kmer_zscore_mat : np.ndarray
        A (num_kmer, num_exp) matrix containing k-mer Z-scores for a single
        probe set.
    """

    # Initialize the matrix
    num_kmer = kmer_count_mat.shape[1]
    num_exp = probe_zscore_mat.shape[1]

    # Step 1 | Replace negative Z-scores with 0
    probe_zscore_mat = np.maximum(probe_zscore_mat, 0)

    # Step 2 | Compute trimmed means for each k-mer across all experiments
    def compute_single_zscore(kmer_idx: int) -> np.ndarray:
        probe_idx_list = find(kmer_count_mat[:, kmer_idx])[0]
        if probe_idx_list.size == 0:

            return np.full(num_exp, np.nan)
        probe_zscore_list = probe_zscore_mat[probe_idx_list, :]

        return np.apply_along_axis(
            trimmed_mean, 0, probe_zscore_list, trim_fraction
        )

    kmer_zscore_list_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_zscore)(kmer_idx) for kmer_idx in range(
            num_kmer)
    )
    kmer_zscore_mat = np.vstack(kmer_zscore_list_list)

    # Step 3 | Normalize across k-mers
    mean_list = np.nanmean(kmer_zscore_mat, axis=0, keepdims=True)
    std_list = np.nanstd(kmer_zscore_mat, axis=0, keepdims=True)
    kmer_zscore_mat = (kmer_zscore_mat - mean_list) / std_list

    return kmer_zscore_mat


def compute_kmer_zscore(
        probe_zscore_df: pd.DataFrame,
        probe_metadata_df: pd.DataFrame,
        trim_fraction: float,
        k: int,
        n_jobs: int
) -> pd.DataFrame:
    """Compute k-mer Z-scores from probe Z-scores for probes in set A, B, and
    AB.

    Parameters
    ----------
    probe_zscore_df : pd.DataFrame
        A (num_probe, num_exp) dataframe containing probe Z-scores.
    probe_metadata_df : pd.DataFrame
        A (num_probe, 2) dataframe containing probe sequences and their set
        assignments.
    trim_fraction : float
        Fraction of probe Z-scores to discard from both ends when computing
        trimmed means.
    k : int
        Length of the k-mers.
    n_jobs: int
        Number of parallel jobs to run.

    Returns
    -------
    kmer_zscore_df : pd.DataFrame
        A (num_kmer, 3 * num_exp) dataframe containing k-mer Z-scores for
        sets A, B, and AB.
    """

    # Count k-mers in probe sequences
    kmer_list, kmer_count_mat = compute_kmer_count(
        probe_metadata_df, k,n_jobs
    )

    # Split probe Z-scores by set
    mask_a_list = probe_metadata_df['set'] == 'SetA'
    mask_b_list = probe_metadata_df['set'] == 'SetB'
    probe_zscore_a_mat = probe_zscore_df.loc[mask_a_list].to_numpy()
    probe_zscore_b_mat = probe_zscore_df.loc[mask_b_list].to_numpy()
    probe_zscore_ab_mat = probe_zscore_df.to_numpy()

    # Split k-mer counts by set
    kmer_count_a_mat = kmer_count_mat[mask_a_list]
    kmer_count_b_mat = kmer_count_mat[mask_b_list]
    kmer_count_ab_mat = kmer_count_mat

    # Compute k-mer Z-scores
    kmer_zscore_a_mat = compute_kmer_zscore_set(
        probe_zscore_a_mat, kmer_count_a_mat, trim_fraction, n_jobs
    )
    kmer_zscore_b_mat = compute_kmer_zscore_set(
        probe_zscore_b_mat, kmer_count_b_mat, trim_fraction, n_jobs
    )
    kmer_zscore_ab_mat = compute_kmer_zscore_set(
        probe_zscore_ab_mat, kmer_count_ab_mat, trim_fraction, n_jobs
    )

    # Combine results
    num_exp = probe_zscore_df.shape[1]
    kmer_zscore_mat = np.empty((len(kmer_list), 3 * num_exp))
    col_list = []
    for i, hyb_id in enumerate(probe_zscore_df.columns):
        kmer_zscore_mat[:, 3 * i] = kmer_zscore_a_mat[:, i]
        kmer_zscore_mat[:, 3 * i + 1] = kmer_zscore_b_mat[:, i]
        kmer_zscore_mat[:, 3 * i + 2] = kmer_zscore_ab_mat[:, i]
        col_list += [f'{hyb_id}_a', f'{hyb_id}_b', f'{hyb_id}_ab']

    # Convert the matrix to dataframe
    kmer_zscore_df = pd.DataFrame(kmer_zscore_mat,
                                  index=kmer_list, columns=col_list)

    return kmer_zscore_df
