"""
summarize.py

Summarize RNAcompete results from multiple experiments.
"""
import glob
import logging
import os
import shutil
from typing import Dict

import pandas as pd

from ..subscripts.summarize_utils import generate_index, generate_subpage

logger = logging.getLogger(__name__)


def run_summarize(root: str,
                  output_path: str,
                  no_html: bool = False,
                  summarize_all: bool = False,
                  summarize_batch: str = None,
                  summarize_experiments: str = None,
                  save_results: bool = True,
                  return_results: bool = True,
    ) -> Dict:
    """
    Entry point for summarizing RNAcompete results from multiple experiments.

    Parameters
    ----------
    root : str
        Root directory containing batch folders.
    output_path : str
        Directory to save summarized results and optional HTML reports.
    no_html : bool
        Combine and save summary tables only.
    summarize_all : bool
        Summarize all experiments across all batches in the root directory.
    summarize_batch : str
        Summarize all experiments within a single batch folder (provide
        batch name).
    summarize_experiments : str
        Summarize selected experiments across all batches in the root
        directory (provide path to a file containing experiment IDs, one per
        line).
    save_results : bool
        Whether to save results or not (default: True).
    return_results : bool
        Whether to return results or not (default: True).

    Returns
    -------
    res_dict : Dict
        Dictionary containing results from computation containing the
        following keys:
        (1) feature_df : Classifier features
        (2) summary_df: Output summary.
    """

    # Validate inputs
    if sum(bool(x) for x in (summarize_all, summarize_batch,
                             summarize_experiments)) != 1:
        raise ValueError('Exactly one of summarize_all, summarize_batch, or '
                         'summarize_experiments must be specified.')

    # Create the output folders
    if save_results:
        os.makedirs(output_path, exist_ok=True)
        if not no_html:
            os.makedirs(os.path.join(output_path, 'html'), exist_ok=True)

    # Get the list of batches
    batch_list = [summarize_batch] if summarize_batch else os.listdir(root)
    batch_list = list(set(batch_list) - {'rnacompete_metadata.tsv'})
    logger.info(f'Selecting experiments within {len(batch_list)} '
                 f'batch(es)...')

    # Get the list of selected experiments
    selected_hyb_id_list = []
    if summarize_experiments is not None:
        with open(summarize_experiments) as f:
            selected_hyb_id_list = f.read().splitlines()

    # Iterate over the batches
    feature_df_list = []
    summary_df_list = []
    for batch in sorted(batch_list):
        batch_folder = os.path.join(root, batch)
        logo_list = glob.glob(os.path.join(batch_folder, 'logo', '*ab.png'))
        hyb_id_list = sorted([os.path.basename(e).split('_')[0]
                              for e in logo_list])

        # Get the selected experiments
        if selected_hyb_id_list:
            hyb_id_list = sorted(list(set(hyb_id_list).intersection(
                set(selected_hyb_id_list))))

        # Read the summary
        if hyb_id_list:
            feature_df = pd.read_csv(os.path.join(
                root, batch, 'feature.tsv'), sep='\t', index_col=0)
            summary_df = pd.read_csv(os.path.join(
                root, batch, 'summary.tsv'), sep='\t', index_col=0)
        else:
            continue

        # Save HTMLs
        feature_df_list.append(feature_df.loc[hyb_id_list])
        summary_df_list.append(summary_df.loc[hyb_id_list])
        if not no_html and save_results:

            # Iterate over the experiments
            for hyb_id in hyb_id_list:
                summary_list = summary_df.loc[hyb_id]

                # Create the folder for the experiment
                hyb_folder = os.path.join(output_path, 'html', hyb_id)
                os.makedirs(hyb_folder, exist_ok=True)

                # Copy the logos and scatter plots
                shutil.copyfile(os.path.join(batch_folder, 'logo',
                                             f'{hyb_id}_a.png'),
                                os.path.join(hyb_folder, 'logo_a.png'))
                shutil.copyfile(os.path.join(batch_folder, 'logo',
                                             f'{hyb_id}_b.png'),
                                os.path.join(hyb_folder, 'logo_b.png'))
                shutil.copyfile(os.path.join(batch_folder, 'logo',
                                             f'{hyb_id}_ab.png'),
                                os.path.join(hyb_folder, 'logo_ab.png'))
                shutil.copyfile(os.path.join(batch_folder, 'scatter',
                                             f'{hyb_id}.png'),
                                os.path.join(hyb_folder, 'scatter.png'))

                # Generate the HTML file
                subpage_path = os.path.join(hyb_folder, f'{hyb_id}.html')
                generate_subpage(hyb_id, summary_list, subpage_path)
        logger.info(f'\tSelected {len(hyb_id_list)} experiments from batch '
                    f'{batch}.')

    # Generate the index file
    feature_df = pd.concat(feature_df_list, axis=0) \
        if feature_df_list else pd.DataFrame()
    summary_df = pd.concat(summary_df_list, axis=0) \
        if summary_df_list else pd.DataFrame()
    if not no_html and save_results:
        logger.info(f'Generating the index HTML file...')
        index_path = os.path.join(output_path, 'html', 'index.html')
        generate_index(summary_df, index_path)

    # Save the combined summary
    if save_results:
        feature_df.to_csv(os.path.join(output_path, 'feature.tsv'), sep='\t')
        summary_df.to_csv(os.path.join(output_path, 'summary.tsv'), sep='\t')

    logger.info('Done.')

    # Return results only when called via API
    if return_results:
        res_dict = {
            'feature_df': feature_df,
            'summary_df': summary_df,
        }
        return res_dict
