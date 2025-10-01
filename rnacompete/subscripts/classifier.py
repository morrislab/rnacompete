"""
classifier.py

Classify experiments as success or failure based on k-mer Z-scores and PWM
features.

Features include:
(1) Pearson correlation coefficient between set A and B 7-mer Z-scores
(2) Number of overlapping top 10 7-mers in set A and B
(3) Z-scores of the top 10 7-mers in set A and B
(4) Skewness of the 7-mer Z-scores in set A and B
(5) Kurtosis of the 7-mer Z-scores in set A and B
(6) Count of each of the 26 artifacts across the top 10 7-mers in set A and B
(7) Sum of (6) across all 26 artifacts
(8) Information content of the PWMs in set A and B
(9) Similarity between the set A and B PWMs
(10) Highest 7-mer Z-score across in set AB
"""

import os
import subprocess
from typing import List, Tuple

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy import stats

from ..classifier import load_classifier


def run_tomtom(pwm_folder: str, hyb_id: str, set_1: str, set_2: str) -> float:
    """
    Compute the similarity between two PWMs using TOMTOM.

    Parameters
    ----------
    pwm_folder : str
        Folder containing the PWMs.
    hyb_id : str
        HybID.
    set_1 : str
        First probe set ('a' or 'b').
    set_2 : str
        Second probe set ('a' or 'b').

    Returns
    -------
    sim : float
        Negative log10(p-value).
    """

    # Get the PWM paths
    pwm_1_path = os.path.join(pwm_folder, f'{hyb_id}_{set_1}.txt')
    pwm_2_path = os.path.join(pwm_folder, f'{hyb_id}_{set_2}.txt')

    # Run TOMTOM
    cmd_1_list = ['tomtom', '-min-overlap', '4', '-norc', '-text', '-thresh',
                  '1', pwm_1_path, pwm_2_path]
    cmd_2_list = ['sed', '-n', '2p']
    ps = subprocess.Popen(cmd_1_list, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    output = subprocess.check_output(cmd_2_list, stdin=ps.stdout).decode()
    sim = -np.log10(float(str(output).split('\t')[3]))

    return sim


def generate_feature(
        kmer_zscore_df: pd.DataFrame,
        top_kmer_mat: np.ndarray,
        top_zscore_mat: np.ndarray,
        pwm_list: List[np.ndarray],
        pwm_folder: str,
        n_jobs: int
) -> pd.DataFrame:
    """
    Generate the classifier features from k-mer Z-scores, top k-mers, and
    PWMs.

    Parameters
    ----------
    kmer_zscore_df : pd.DataFrame
        A (num_kmer, 3 * num_exp) dataframe containing k-mer Z-scores for
        sets A, B, and AB.
    top_kmer_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top k-mers for all
        experiments and probe sets (A, B, AB).
    top_zscore_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top k-mer Z-scores for
        all experiments and probe sets (A, B, AB).
    pwm_list : List[np.ndarray]
        List of PWMs, each of shape (num_pos, 4), with columns corresponding
        to A, C, G, and U.
    pwm_folder : str
        Folder containing the PWMs.
    n_jobs : int
        Number of parallel jobs to run.

    Returns
    -------
    feature_df : pd.DataFrame
        A (num_exp, 58) dataframe containing the 58 feature values across all
        experiments.
    """

    # Define the list of artifacts
    artifact_list = ['AAAUAAA', 'AGACCC', 'AGACGGG', 'CCCC', 'CGGAGG',
                     'GACUCAC', 'GACUGCC', 'GAGUC', 'GGGAGC', 'GGGG',
                     'GGGGAGC', 'GGGGCC', 'GGGCC', 'GGGGGGG', 'AGGCCC',
                     'AGGGCC', 'UUUUUUU', 'AGGGC', 'CCCCC', 'CCCCCC',
                     'GGGGG', 'GGGGGG', 'CCCCCCC', 'GGCGG', 'AGAGCC',
                     'GUCGU']
    hyb_id_list = np.unique([e.split('_')[0] for e in kmer_zscore_df.columns])

    # Process experiments in parallel
    def compute_single(idx: int, hyb_id: str) -> List[float]:

        # Get the Z-scores
        zscore_a_list = kmer_zscore_df[f'{hyb_id}_a'].dropna()
        zscore_b_list = kmer_zscore_df[f'{hyb_id}_b'].dropna()

        # Get the top k-mers
        top_kmer_a_list = top_kmer_mat[3 * idx]
        top_kmer_b_list = top_kmer_mat[3 * idx + 1]
        top_kmer_list = list(top_kmer_a_list) + list(top_kmer_b_list)

        # Get the top Z-scores
        top_zscore_a_list = top_zscore_mat[3 * idx]
        top_zscore_b_list = top_zscore_mat[3 * idx + 1]
        top_zscore_ab_list = top_zscore_mat[3 * idx + 2]

        # Get the PWMs
        pwm_a_mat = pwm_list[3 * idx]
        pwm_b_mat = pwm_list[3 * idx + 1]

        # Compute the Z-score statistics
        pearson = stats.pearsonr(zscore_a_list, zscore_b_list)[0]
        num_intersect = len(set(top_kmer_a_list)
                            .intersection(set(top_kmer_b_list)))
        skew_a = stats.skew(zscore_a_list)
        skew_b = stats.skew(zscore_b_list)
        kurt_a = stats.kurtosis(zscore_a_list, fisher=False)
        kurt_b = stats.kurtosis(zscore_b_list, fisher=False)
        artifact_count_list = [np.sum([artifact in e for e in top_kmer_list])
                               for artifact in artifact_list]
        artifact_count_sum = sum(artifact_count_list)

        # Compute the information content of the PWMs
        ic_a = 14 + (pwm_a_mat * np.log2(pwm_a_mat)).sum()
        ic_b = 14 + (pwm_b_mat * np.log2(pwm_b_mat)).sum()

        # Compute the motif similarity
        motif_sim_ab = run_tomtom(pwm_folder, hyb_id, 'a', 'b')
        motif_sim_ba = run_tomtom(pwm_folder, hyb_id, 'b', 'a')

        # Combine features
        feature_list = [pearson, num_intersect,
                        *top_zscore_a_list, *top_zscore_b_list,
                        skew_a, skew_b, kurt_a, kurt_b,
                        *artifact_count_list, artifact_count_sum,
                        ic_a, ic_b, motif_sim_ab, motif_sim_ba,
                        top_zscore_ab_list[0]]

        return feature_list

    feature_list_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_single)(idx, hyb_id)
        for idx, hyb_id in enumerate(hyb_id_list)
    )
    feature_mat = np.array(feature_list_list)

    # Convert matrix into DataFrame
    col_list = ['zscoreCorrelation', 'top10overlap',
                'aZ1', 'aZ2', 'aZ3', 'aZ4', 'aZ5',
                'aZ6', 'aZ7', 'aZ8', 'aZ9', 'aZ10',
                'bZ1', 'bZ2', 'bZ3', 'bZ4', 'bZ5',
                'bZ6', 'bZ7', 'bZ8', 'bZ9', 'bZ10',
                'aSkewness', 'bSkewness', 'aKurtosis', 'bKurtosis',
                *artifact_list, 'total_artifacts',
                'aIC', 'bIC', 'AtoB', 'BtoA', 'setAB.zscore']
    feature_df = pd.DataFrame(feature_mat, index=hyb_id_list,
                              columns=col_list)
    feature_df.index.name = 'hyb_id'

    return feature_df


def run_classifier(
        kmer_zscore_df: pd.DataFrame,
        top_kmer_mat: np.ndarray,
        top_zscore_mat: np.ndarray,
        pwm_list: List[np.ndarray],
        pwm_folder: str,
        n_jobs: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Classify experiments as success or failure based on k-mer Z-scores and PWM
    features.

    Parameters
    ----------
    kmer_zscore_df : pd.DataFrame
        A (num_kmer, 3 * num_exp) dataframe containing k-mer Z-scores for
        sets A, B, and AB.
    top_kmer_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top k-mers for all
        experiments and probe sets (A, B, AB).
    top_zscore_mat : np.ndarray
        A (3 * num_exp, top_k) matrix containing the top k-mer Z-scores for
        all experiments and probe sets (A, B, AB).
    pwm_list : List[np.ndarray]
        List of PWMs, each of shape (num_pos, 4), with columns corresponding
        to A, C, G, and U.
    pwm_folder : str
        Folder containing the PWMs.
    n_jobs : int
        Number of parallel jobs to run.

    Returns
    -------
    feature_df : pd.DataFrame
        A (num_exp, 58) dataframe containing the 58 feature values across all
        experiments.
    prob_list : np.ndarray
        Array of success probabilities for each experiment.
    """

    # Get the features
    feature_df = generate_feature(
        kmer_zscore_df, top_kmer_mat, top_zscore_mat, pwm_list, pwm_folder,
        n_jobs
    )

    # Load the model
    ss, lr = load_classifier()

    # Run the model
    feature_mat = ss.transform(feature_df.to_numpy())
    prob_list = lr.predict_proba(feature_mat)[:, 1]

    return feature_df, prob_list
