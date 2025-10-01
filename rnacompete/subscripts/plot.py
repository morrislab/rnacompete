"""
plot.py

Plot and save scatter plots showing k-mer Z-score correlations between
set A and set B probes across all experiments.

Plot and save sequence logos for all experiments and sets.
"""
import os
from typing import List

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_scatter(kmer_zscore_df: pd.DataFrame, scatter_folder: str) -> None:
    """
    Plot and save scatter plots showing k-mer Z-score correlations between
    set A and set B probes across all experiments.

    Parameters
    ----------
    kmer_zscore_df : pd.DataFrame
        A (num_kmer, 3 * num_exp) dataframe containing k-mer Z-scores for
        sets A, B, and AB.
    scatter_folder : str
        Folder to save the scatter plots.
    """
    os.makedirs(scatter_folder, exist_ok=True)

    # Iterate over the experiments
    for hyb_id in [e.split('_')[0] for e in kmer_zscore_df.columns[::3]]:

        # Get the Z-scores
        zscore_a_list = kmer_zscore_df[f'{hyb_id}_a'].to_list()
        zscore_b_list = kmer_zscore_df[f'{hyb_id}_b'].to_list()

        # Generate the plot
        plt.figure(figsize=(4, 4))
        plt.scatter(zscore_a_list, zscore_b_list, color='black', s=5,
                    zorder=3)

        # Set the limits
        lim_min = min(zscore_a_list + zscore_b_list) - 1
        lim_max = max(zscore_a_list + zscore_b_list)
        lim_max = np.ceil(lim_max / 5) * 5
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)

        # Set the ticks
        plt.xticks(np.arange(0, lim_max + 5, 5))
        plt.yticks(np.arange(0, lim_max + 5, 5))

        # Set the labels
        plt.xlabel('Set A', weight='bold')
        plt.ylabel('Set B', weight='bold')
        plt.grid(color='lightgray', zorder=0)

        # Save the figure
        scatter_path = os.path.join(scatter_folder, f'{hyb_id}.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_logo(hyb_id_set_list: List[str], pwm_list: List[np.ndarray],
              logo_folder: str) -> None:
    """
    Plot and save sequence logos for all experiments and sets.

    Parameters
    ----------
    hyb_id_set_list : List[str]
        List of the Hyb IDs and their sets (e.g., HybID00001_a, HybID00001_b,
        HybID00001_ab).
    pwm_list : List[np.ndarray]
        List of PWMs, each of shape (num_pos, 4), with columns corresponding
        to A, C, G, and U.
    logo_folder : str
        Folder to save the sequence logos.
    """
    os.makedirs(logo_folder, exist_ok=True)

    # Define the color scheme
    color_scheme = {
        'A': '#00CC00',
        'C': '#0000CC',
        'G': '#FFB302',
        'U': '#CC0001',
    }

    # Iterate over the PWMs
    for hyb_id_set, pwm in zip(hyb_id_set_list, pwm_list):
        # Convert per-position information content (in bits)
        bit_list = np.clip(2 + np.sum(pwm * np.log2(pwm), axis=1), 0, None)
        logo_mat = pwm * bit_list[:, None]

        # Plot the logo
        logomaker.Logo(pd.DataFrame(logo_mat, columns=['A', 'C', 'G', 'U']),
                       color_scheme=color_scheme,
                       show_spines=False)
        plt.xticks([])
        plt.yticks([])

        # Save the logo
        logo_path = os.path.join(logo_folder, f'{hyb_id_set}.png')
        plt.savefig(logo_path, dpi=300, bbox_inches='tight')
        plt.close()
