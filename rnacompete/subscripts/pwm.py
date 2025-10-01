"""
pwm.py

Compute, save, and load the position weight matrices (PWMs) across all
experiments and sets.

The computation proceeds as follows:
(1) Compute PFMs by counting nucleotide frequencies at each aligned position
    across top k-mers.
(2) Compute PWMs as weighted averages of Z-scores per nucleotide per position.
(3) Trim the PWMs to retain center positions where any nucleotide appears in
    more than 50% of k-mers.
"""

from typing import List

from joblib import Parallel, delayed
import numpy as np


def compute_pwm(aligned_top_kmer_mat: np.ndarray,
                top_zscore_mat: np.ndarray,
                n_jobs: int) -> List[np.ndarray]:
    """
    Compute and save PWMs from top aligned k-mers and their Z-scores.

    Parameters
    ----------
    aligned_top_kmer_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the aligned k-mers padded
        with '-' for consistent length.
    top_zscore_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top k-mer Z-scores for
        all experiments and probe sets (A, B, AB).
    n_jobs : int
        Number of parallel jobs to run.

    Returns
    -------
    pwm_list : List[np.ndarray]
        List of PWMs, each of shape (num_pos, 4), with columns corresponding
        to A, C, G, and U.
    """

    # Compute the PWMs in parallel
    nt_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    def compute_single(
            kmer_list: np.ndarray, zscore_list: np.ndarray
    ) -> np.ndarray:

        # Initialize the variables
        mean_zscore = np.mean(zscore_list)
        num_pos = len(kmer_list[0])
        pfm = np.zeros((num_pos, 4))
        pwm = np.zeros((num_pos, 4))
        kmer_array = np.array([list(kmer) for kmer in kmer_list])

        for nt, idx in nt_map.items():
            mask = (kmer_array == nt)
            pfm[:, idx] = mask.sum(axis=0)
            pwm[:, idx] = (mask * zscore_list[:, None]).sum(axis=0)

        # Handle gaps
        gap_mask = (kmer_array == '-')
        pwm += (gap_mask.sum(axis=0)[:, None] * (mean_zscore / 4))

        # Trim the PWM
        coverage = pfm.sum(axis=1) / len(kmer_list)
        start = np.argmax(coverage >= 0.5)
        end = len(coverage) - np.argmax(coverage[::-1] >= 0.5)
        pwm = pwm[start:end]

        # Add pseudo-count and convert to fraction
        pwm += 1
        pwm /= pwm.sum(axis=1)[:, None]

        return pwm

    pwm_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_single)(kmer_list, zscore_list)
        for kmer_list, zscore_list in zip(
            aligned_top_kmer_mat, top_zscore_mat
        )
    )

    return pwm_list


def save_pwm(hyb_id_set: str, pwm: np.ndarray, pwm_path: str) -> None:
    """
    Save a position weight matrix (PWM) in MEME format.

    Parameters
    ----------
    hyb_id_set : str
        Hyb ID and set.
    pwm : np.ndarray
        PWM of shape (L, 4), where L is the motif length and columns
        correspond to A, C, G, and U.
    pwm_path : str
        Path to the output MEME file.
    """

    # Write the header
    line_list = [
        'MEME version 5.5.8\n\n',
        'ALPHABET= ACGU\n\n',
        'Background letter frequencies (from uniform background):\n',
        'A 0.25000 C 0.25000 G 0.25000 U 0.25000 \n\n',
        f'MOTIF {hyb_id_set}\n\n',
        f'letter-probability matrix:\n'
    ]

    # Write the PWM
    for pwm_list in pwm:
        line_list.append(
            '  ' + '\t'.join(f'{val:.6f}' for val in pwm_list) + '\n'
        )

    # Save the file
    with open(pwm_path, 'w') as f:
        f.writelines(line_list)


def load_pwm(pwm_path: str) -> np.ndarray:
    """
    Load a position weight matrix (PWM) in MEME format.

    Parameters
    ----------
    pwm_path : str
        Path to the MEME file.

    Returns
    -------
    pwm : np.ndarray
        PWM of shape (L, 4), where L is the motif length and columns
        correspond to A, C, G, and U.
    """

    line_list = []
    with open(pwm_path) as f:
        capture = False
        for line in f:
            if line.startswith('letter-probability matrix'):
                capture = True
                continue
            if capture:
                if line.strip() == '':
                    break
                line_list.append([float(x) for x in line.strip().split()])
    pwm = np.array(line_list)

    return pwm
