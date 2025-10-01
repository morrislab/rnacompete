"""
summarize_utils.py

Utilities for the summarize mode.
"""

import numpy as np
import pandas as pd


def generate_subpage(hyb_id: str, summary_list: pd.Series,
                     subpage_path: str) -> None:
    """
    Generate the subpage HTML file.

    Parameters
    ----------
    hyb_id : str
        Hyb ID.
    summary_list : pd.Series
        Summary of the experiment.
    subpage_path : str
        Path to the HTML subpage file.
    """

    # Read the summary
    rnacompete_id = summary_list['rnacompete_id']
    tax_name = summary_list['tax_name']
    gene_name = summary_list['gene_name']
    cls = summary_list['class']
    prob = summary_list['prob']

    # Write the header
    html_str = (f'<h1>{hyb_id} | {rnacompete_id} | {tax_name} | {gene_name} |'
                f' {cls} ({prob:.2f})</h1>\n')

    # Write the scatter plot
    html_str += '<h2>7-mer Scatter plot</h2>\n'
    html_str += ('<img src="scatter.png" style="max-width:400px; '
                 'height:auto;">\n')
    html_str += '<hr>\n'

    # Write the logos
    logo_df = pd.DataFrame({'Set A': ['<img src="logo_a.png" height=80>'],
                            'Set B': ['<img src="logo_b.png" height=80>'],
                            'Set AB': ['<img src="logo_ab.png" height=80>']})
    html_str += '<h2>Motifs</h2>\n'
    html_str += logo_df.to_html(index=False, justify='center', escape=False)
    html_str += '<hr>\n'

    # Write the 7-mers and Z-scores
    kmer_col_list = summary_list.index[5:95]
    summary_mat = summary_list[kmer_col_list].to_numpy().reshape(-1, 10)

    # Wrap each element in <tt>
    summary_mat = np.char.add('<tt>', summary_mat.astype(str))
    summary_mat = np.char.add(summary_mat, '</tt>')

    # Record rows for display
    kmer_mat = np.vstack((
        summary_mat[[0, 3, 6]],
        ['⬛⬛'] * 10,
        summary_mat[[1, 4, 7]],
        ['⬛⬛'] * 10,
        summary_mat[[2, 5, 8]],
        ['⬛⬛'] * 10
    )).T
    col_list = ['Set A 7-mer', 'Set A Z-score', 'Set A Aligned', '⬛⬛',
                'Set B 7-mer', 'Set B Z-score', 'Set B Aligned', '⬛⬛',
                'Set AB 7-mer', 'Set AB Z-score', 'Set AB Aligned', '⬛⬛', ]
    kmer_df = pd.DataFrame(kmer_mat, columns=col_list)
    html_str += '<h2>Top 7-mers and Z-scores</h2>\n'
    html_str += kmer_df.to_html(
        index=False, na_rep='', justify='center', escape=False
    )
    html_str += '<hr>\n'
    html_str += '<a href="../index.html">Go back</a>'

    # Write to file
    with open(subpage_path, 'w') as f:
        f.write(html_str)


def generate_index(summary_df: pd.DataFrame, index_path: str) -> None:
    """
    Generate the index HTML file.

    Parameters
    ----------
    summary_df : pd.DataFrame
        A (num_exp, 96) dataframe containing the summary across all
        experiments.
    index_path : str
        Path to the index HTML file.
    """

    # Generate the table
    index_df = pd.DataFrame()
    index_df['Hyb ID'] = [
        f'<a href="{hyb_id}/{hyb_id}.html">{hyb_id}</a>'
        for hyb_id in summary_df.index
    ]
    index_df['RNAcompete ID'] = summary_df['rnacompete_id'].to_list()
    index_df['Organism'] = summary_df['tax_name'].to_list()
    index_df['Gene name'] = summary_df['gene_name'].to_list()

    # Color-coded classification
    classifier_list = [
        f'<span style="color:red">{cls} ({prob:.2f})</span>'
        if cls == 'Failure'
        else f'<span style="color:green">{cls} ({prob:.2f})</span>'
        for cls, prob in zip(summary_df['class'], summary_df['prob'])
    ]
    index_df['Classification'] = classifier_list

    # Logos
    index_df['Logo'] = [
        f'<img src="{hyb_id}/logo_ab.png" height=80>'
        for hyb_id in summary_df.index]

    # Convert the table to HTML
    html_str = index_df.to_html(
        index=False, na_rep='', justify='center', escape=False
    )
    with open(index_path, 'w') as f:
        f.write(html_str)
