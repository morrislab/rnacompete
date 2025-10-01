"""
process_utils.py

Utilities for the process mode.
"""

from typing import List, Tuple

from joblib import Parallel, delayed
import numpy as np
import pandas as pd


def get_top_kmer(kmer_zscore_df: pd.DataFrame, top_k: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the top k-mers and Z-scores for all experiments and sets.

    Parameters
    ----------
    kmer_zscore_df : pd.DataFrame
        A (num_kmer, 3 * num_exp) dataframe containing k-mer Z-scores for
        sets A, B, and AB.
    top_k : int
        Number of top k-mers used to generate the motif.

    Returns
    -------
    top_kmer_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top k-mers for all
        experiments and probe sets (A, B, AB).
    top_zscore_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top k-mer Z-scores for
        all experiments and probe sets (A, B, AB).
    """

    # Iterate over the experiments and sets
    top_kmer_mat = []
    top_zscore_mat = []
    for hyb_id_set in kmer_zscore_df.columns:

        # Get the top k-mers and Z-scores
        tmp_df = kmer_zscore_df[hyb_id_set].sort_values(
            ascending=False)[:top_k]
        top_kmer_mat.append(tmp_df.index.to_list())
        top_zscore_mat.append(tmp_df.to_list())
    top_kmer_mat = np.array(top_kmer_mat)
    top_zscore_mat = np.array(top_zscore_mat)
    return top_kmer_mat, top_zscore_mat


def align(kmer_1: str, kmer_2: str) -> Tuple[int, int]:
    """
    Align two k-mers to maximize matching nucleotides.

    The two k-mers are shifted left and right against each other to find
    the optimal offset.

    Parameters
    ----------
    kmer_1 : str
        First k-mer.
    kmer_2 : str
        Second k-mer.

    Returns
    -------
    max_match : int
        Maximum number of matching nucleotides.
    min_offset : int
        Offset required for maximum matching. Negative offset indicate that
        kmer_2 is shifted left.
    """
    max_match = 0
    min_offset = 0
    k = len(kmer_1)

    # Shift kmer_1 left
    for i in range(k):
        match = sum(a == b for a, b in zip(kmer_1[i:], kmer_2))
        if match > max_match:
            max_match = match
            min_offset = i

    # Shift kmer_2 left
    for i in range(k):
        match = sum(a == b for a, b in zip(kmer_2[i:], kmer_1))
        if match > max_match:
            max_match = match
            min_offset = -i

    return max_match, min_offset


def align_top_kmer(top_kmer_mat: np.ndarray, n_jobs: int) -> np.ndarray:
    """
    Align all k-mers in each experiment/set to the seed k-mer.

    The seed k-mer is chosen as the one with the most average matches against
    other k-mers and the least average offset.

    Parameters
    ----------
    top_kmer_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top k-mers for all
        experiments and probe sets (A, B, AB).
    n_jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
    aligned_top_kmer_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the aligned k-mers padded
        with '-' for consistent length.
    """

    # Perform alignment for experiments in parallel
    def compute_single(kmer_list: List[str]) -> List[str]:

        # Find the seed k-mer
        top_k = len(kmer_list)
        k = len(kmer_list[0])
        match_sum_list = np.zeros(top_k, dtype=int)
        offset_sum_list = np.zeros(top_k, dtype=int)

        # Iterate over the k-mers
        for i, kmer_1 in enumerate(kmer_list):
            match_sum, offset_sum = 0, 0

            # Iterate over the k-mers
            for kmer_2 in kmer_list:

                # Perform alignment and get total sum and offset
                match, offset = align(kmer_1, kmer_2)
                match_sum += match
                offset_sum += abs(offset)
            match_sum_list[i] = match_sum
            offset_sum_list[i] = offset_sum

        # Select the seed k-mer
        best_idx_list = np.flatnonzero(match_sum_list == match_sum_list.max())
        best_idx = best_idx_list[np.argmin(offset_sum_list[best_idx_list])]

        # Perform the alignment using the seed k-mer
        offset_list = np.array([
            align(kmer_list[best_idx], kmer)[1] for kmer in kmer_list
        ])
        min_offset, max_offset = offset_list.min(), offset_list.max()
        total_len = k + (max_offset - min_offset)

        # Format the alignments
        aligned_list = np.full((top_k, total_len), '-', dtype='<U1')
        for i, (kmer, offset) in enumerate(zip(kmer_list, offset_list)):
            start = offset - min_offset
            aligned_list[i, start:start + k] = list(kmer)

        return [''.join(aligned) for aligned in aligned_list]

    aligned_top_kmer_list_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_single)(kmer_list) for kmer_list in top_kmer_mat
    )
    aligned_top_kmer_mat = np.array(aligned_top_kmer_list_list)

    return aligned_top_kmer_mat


def get_summary(batch_metadata_df, top_kmer_mat, top_zscore_mat,
                aligned_top_kmer_mat, class_list, prob_list):
    """
    Generate a summary dataframe containing the metadata, classifier output,
    top k-mers, Z-scores, and aligned k-mers.

    Parameters
    ----------
    batch_metadata_df : pd.DataFrame
        A (num_exp, 4) dataframe containing the RNAcompete ID, taxa name,
        gene name, and batch name of all experiments in the batch.
    top_kmer_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top k-mers for all
        experiments, using probes in set A, set B, and set AB.
    top_zscore_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top k-mer Z-scores for
        all experiments, using probes in set A, set B, and set AB.
    aligned_top_kmer_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top aligned k-mers for
        all experiments, using probes in set A, set B, and set AB.
    class_list : list
        List of the predicted classes (success or failure) across all
        experiments.
    prob_list : np.array
        List of the success probabilities across all experiments.

    Returns
    -------
    summary_df : pd.DataFrame
        A (num_exp, 96) dataframe containing the summary across all
        experiments.
    """

    # Get the column names
    col_list = ['rnacompete_id', 'tax_name', 'gene_name', 'class', 'prob']
    for prefix in ['kmer', 'zscore', 'aligned_kmer']:
        for set_name in ['a', 'b', 'ab']:
            for idx in range(1, 11):
                col_list += [f'{prefix}_{set_name}_{idx}']

    # Format the data
    rnacompete_id_list = np.array(
        [batch_metadata_df['rnacompete_id'].to_list()]
    ).T
    tax_name_list = np.array([batch_metadata_df['tax_name'].to_list()]).T
    gene_name_list = np.array([batch_metadata_df['gene_name'].to_list()]).T
    class_list = np.array([class_list]).T
    prob_list = np.array([prob_list]).T
    kmer_mat = top_kmer_mat.reshape(len(top_kmer_mat) // 3, -1)
    zscore_mat = top_zscore_mat.reshape(len(top_zscore_mat) // 3, -1)
    aligned_kmer_mat = aligned_top_kmer_mat.reshape(
        len(aligned_top_kmer_mat) // 3, -1)
    summary_mat = np.hstack(
        (rnacompete_id_list, tax_name_list, gene_name_list, class_list,
         prob_list,
         kmer_mat, zscore_mat, aligned_kmer_mat))
    summary_df = pd.DataFrame(summary_mat, index=batch_metadata_df.index,
                              columns=col_list)
    summary_df.index.name = 'hyb_id'

    return summary_df
