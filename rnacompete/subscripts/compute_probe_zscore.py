"""
compute_probe_zscore.py

Compute probe Z-scores from probe intensities.

Each row corresponds to a probe, and each column corresponds to an experiment.

The computation proceeds as follows:
(1) Adjust column medians and IQRs to their geometric means across all
    columns.
(2) Normalize each row by subtracting its median and dividing by 1.4826 * MAD.
(3) Normalize each column by subtracting its median and dividing by 1.4826 *
    MAD.
"""

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation


def compute_probe_zscore(
        probe_intensity_df: pd.DataFrame,
        probe_metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute probe Z-scores from probe intensities.

    Parameters
    ----------
    probe_intensity_df : pd.DataFrame
        A (num_probe, 2 * num_exp) dataframe containing the probe intensities.
        Each experiment is represented by two consecutive columns.
            - first column: probe intensities
            - second column: manual flags (non-zero indicates probes to drop)
    probe_metadata_df : pd.DataFrame
        A (num_probe, 2) dataframe containing probe sequences and their set
        assignments.

    Returns
    -------
    probe_zscore_df : pd.DataFrame
        A (num_probe, num_exp) dataframe containing probe Z-scores.
    """

    # Subset for probes in the metadata
    probe_id_list = probe_metadata_df.index
    probe_intensity_df = probe_intensity_df.loc[probe_id_list]

    # Extract intensity and flag matrices
    intensity_mat = probe_intensity_df.iloc[:, ::2].to_numpy()
    flag_mat = probe_intensity_df.iloc[:, 1::2].to_numpy()
    intensity_mat = np.where(flag_mat > 0, np.nan, intensity_mat)

    # Step 1 | Adjust column medians and IQRs
    col_median_list = np.nanmedian(intensity_mat, axis=0)
    col_iqr_list = (
        np.nanquantile(intensity_mat, 0.75, axis=0, method='hazen')
        - np.nanquantile(intensity_mat, 0.25, axis=0, method='hazen')
    )
    step_1_mat = (intensity_mat - col_median_list) / col_iqr_list
    geom_median = np.exp(np.nanmean(np.log(col_median_list)))
    geom_iqr = np.exp(np.nanmean(np.log(col_iqr_list)))
    step_1_mat = step_1_mat * geom_iqr + geom_median

    # Step 2 | Normalize rows by median and MAD
    row_median_list = np.expand_dims(np.nanmedian(step_1_mat, axis=1), axis=1)
    row_mad_list = np.expand_dims(
        median_abs_deviation(step_1_mat, axis=1, nan_policy='omit'), axis=1
    )
    step_2_mat = (step_1_mat - row_median_list) / (1.4826 * row_mad_list)

    # Step 3 | Normalize columns by median and MAD
    col_median_list = np.nanmedian(step_2_mat, axis=0)
    col_mad_list = median_abs_deviation(step_2_mat, axis=0, nan_policy='omit')
    probe_zscore_mat = ((step_2_mat - col_median_list)
                        / (1.4826 * col_mad_list))
    probe_zscore_df = pd.DataFrame(
        probe_zscore_mat,
        index=probe_intensity_df.index,
        columns=probe_intensity_df.columns[::2]
    )
    probe_zscore_df.index.name = None

    return probe_zscore_df
