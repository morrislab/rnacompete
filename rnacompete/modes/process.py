"""
process.py

Process RNAcompete data for a single batch.

The computation proceeds as follows:
(1) Compute the probe Z-scores.
(2) Compute the k-mer Z-scores.
(3) Plot the scatter plots.
(4) Get the top k-mers.
(5) Align the top k-mers.
(6) Compute the PWMs.
(7) Plot the sequence logos.
(8) Run the classifier.
(9) Generate the summary.
"""

import logging
import os
from typing import Dict

import pandas as pd

from ..data import load_probe_metadata
from ..subscripts.compute_probe_zscore import compute_probe_zscore
from ..subscripts.compute_kmer_zscore import compute_kmer_zscore
from ..subscripts.plot import plot_logo, plot_scatter
from ..subscripts.pwm import compute_pwm, load_pwm, save_pwm
from ..subscripts.classifier import run_classifier
from ..subscripts.process_utils import (get_top_kmer, align_top_kmer,
                                        get_summary)

logger = logging.getLogger(__name__)


def run_process(root : str,
                batch : str,
                trim_fraction: float = 0.025,
                k: int = 7,
                top_k: int = 10,
                n_jobs: int = 1,
                save_results: bool = True,
                return_results: bool = True
) -> Dict:
    """
    Entry point for processing RNAcompete data for a single batch.

    Parameters
    ----------
    root : str
        Root directory containing batch folders.
    batch : str
        Name of the batch folder to process.
    trim_fraction : float
        Fraction of probe Z-scores to discard from both ends when computing
        trimmed means (default: 0.025).
    k : int
        Length of the k-mers (default: 7).
    top_k : int
        Number of top k-mers used to generate the motif (default: 10).
    n_jobs : int
        Number of parallel jobs to run (default: 1).
    save_results : bool
        Whether to save results or not (default: True).
    return_results : bool
        Whether to return results or not (default: True).

    Returns
    -------
    res_dict : Dict
        Dictionary containing results from computation containing the
        following keys:
        (1) probe_intensity_df : Probe intensities
        (2) probe_zscore_df: Probe Z-scores
        (3) kmer_zscore_df: K-mer Z-scores
        (4) pwm_list: PWMs.
        (5) feature_df: Classifier features.
        (6) summary_df: Output summary.
    """

    # Get the directories
    batch_folder = os.path.join(root, batch)
    if not os.path.exists(batch_folder):
        raise FileNotFoundError(f'Batch folder not found: {batch_folder}')
    path_dict = {
        'probe_intensity': os.path.join(batch_folder, 'probe_intensity.tsv'),
        'probe_zscore': os.path.join(batch_folder, 'probe_zscore.tsv'),
        'kmer_zscore': os.path.join(batch_folder, 'kmer_zscore.tsv'),
        'scatter': os.path.join(batch_folder, 'scatter'),
        'pwm': os.path.join(batch_folder, 'pwm'),
        'logo': os.path.join(batch_folder, 'logo'),
        'feature': os.path.join(batch_folder, 'feature.tsv'),
        'summary': os.path.join(batch_folder, 'summary.tsv')
    }
    for key in ['scatter', 'pwm', 'logo']:
        os.makedirs(path_dict[key], exist_ok=True)

    # Read the input files
    probe_metadata_df = load_probe_metadata()
    rnacompete_metadata_df = pd.read_csv(
        os.path.join(root, 'rnacompete_metadata.tsv'), sep='\t', index_col=0
    )
    probe_intensity_df = pd.read_csv(
        path_dict['probe_intensity'], sep='\t', index_col=0
    )

    # Step 1 | Compute the probe Z-scores
    logger.info('STEP 1 | Computing the probe Z-scores...')
    if os.path.exists(path_dict['probe_zscore']):
        logger.info('\tLoading existing probe Z-scores...')
        probe_zscore_df = pd.read_csv(
            path_dict['probe_zscore'], sep='\t', index_col=0
        )
    else:
        probe_zscore_df = compute_probe_zscore(
            probe_intensity_df, probe_metadata_df
        )
        if save_results:
            probe_zscore_df.to_csv(path_dict['probe_zscore'], sep='\t')

    # Step 2 | Compute the k-mer Z-scores
    logger.info('STEP 2 | Computing the k-mer Z-scores...')
    if os.path.exists(path_dict['kmer_zscore']):
        logger.info('\tLoading existing k-mer Z-scores...')
        kmer_zscore_df = pd.read_csv(
            path_dict['kmer_zscore'], sep='\t', index_col=0
        )
    else:
        kmer_zscore_df = compute_kmer_zscore(
            probe_zscore_df, probe_metadata_df, trim_fraction, k, n_jobs
        )
        if save_results:
            kmer_zscore_df.to_csv(path_dict['kmer_zscore'], sep='\t')

    # Step 3 | Plot the scatter plot
    logger.info('STEP 3 | Plotting the scatter plots...')
    scatter_path_list = [
        os.path.join(path_dict['scatter'], f'{col}.png') for col
        in probe_zscore_df.columns
    ]
    scatter_exist = all(os.path.exists(scatter_path)
                        for scatter_path in scatter_path_list)
    if scatter_exist:
        logger.info('\tScatter plots already exist.')
    elif save_results:
        plot_scatter(kmer_zscore_df, path_dict['scatter'])

    # Step 4 | Get the top k-mers
    logger.info('STEP 4 | Getting the top k-mers...')
    top_kmer_mat, top_zscore_mat = get_top_kmer(kmer_zscore_df, top_k)

    # Step 5 | Align the top k-mers
    logger.info('STEP 5 | Aligning the top k-mers...')
    aligned_top_kmer_mat = align_top_kmer(top_kmer_mat, n_jobs)

    # Step 6 | Compute the PWMs
    logger.info('STEP 6 | Computing the PWMs...')
    pwm_path_list = [os.path.join(path_dict['pwm'], f'{col}.txt') for col
                     in kmer_zscore_df.columns]
    pwm_exist = all(os.path.exists(pwm_path) for pwm_path in pwm_path_list)
    if pwm_exist:
        logger.info('\tLoading existing PWMs...')
        pwm_list = []
        for pwm_path in pwm_path_list:
            pwm_list.append(load_pwm(pwm_path))
    else:
        pwm_list = compute_pwm(aligned_top_kmer_mat, top_zscore_mat, n_jobs)
        if save_results:
            logger.info('\tSaving the PWMs...')
            for hyb_id_set, pwm in zip(kmer_zscore_df.columns, pwm_list):
                pwm_path = os.path.join(path_dict['pwm'], f'{hyb_id_set}.txt')
                save_pwm(hyb_id_set, pwm, pwm_path)

    # Step 7 | Plot the sequence logos
    logger.info('STEP 7 | Plotting the sequence logos...')
    logo_exist = all(
        os.path.exists(os.path.join(path_dict['logo'], f'{col}.png'))
        for col in kmer_zscore_df.columns
    )
    if logo_exist:
        logger.info('\tSequence logos already exist.')
    elif save_results:
        plot_logo(kmer_zscore_df.columns.to_list(), pwm_list,
                  path_dict['logo'])

    # Step 8 | Run the classifier
    logger.info('STEP 8 | Running the classifier...')
    feature_df, prob_list = run_classifier(
        kmer_zscore_df, top_kmer_mat, top_zscore_mat, pwm_list,
        path_dict['pwm'], n_jobs
    )
    class_list = ['Success' if e >= 0.5 else 'Failure' for e in prob_list]
    if save_results:
        feature_df.to_csv(path_dict['feature'], sep='\t')

    # Step 9 | Generate the summary
    logger.info('STEP 9 | Generating the summary...')
    batch_metadata_df = rnacompete_metadata_df.loc[probe_zscore_df.columns]
    summary_df = get_summary(
        batch_metadata_df, top_kmer_mat, top_zscore_mat,
        aligned_top_kmer_mat, class_list, prob_list
    )
    if save_results:
        summary_df.replace('nan', '').to_csv(
            path_dict['summary'], sep='\t')

    logger.info('Done.')

    # Return results only when called via API
    if return_results:
        res_dict = {
            'probe_intensity_df': probe_intensity_df,
            'probe_zscore_df': probe_zscore_df,
            'kmer_zscore_df': kmer_zscore_df,
            'pwm_list': pwm_list,
            'feature_df': feature_df,
            'summary_df': summary_df
        }
        return res_dict
