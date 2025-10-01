[![DOI](https://zenodo.org/badge/1068117834.svg)](https://doi.org/10.5281/zenodo.17247113)

# RNAcompete: Data Processing and Summarization Pipeline
This repository provides a reproducible pipeline to process and summarize
**RNAcompete** data.

Briefly, the pipeline:

- **Normalizes probe intensities** in batches,
- Converts them into **7-mer Z-scores**,
- Builds **position weight matrices (PWMs)** and **sequence logos**,
- Generates **scatter plots** for quality assessment,
- Applies a **success/failure classifier** to flag problematic experiments,
- Summarizes results into **combined tables** and optional **HTML reports** for
easy review.

## Installation
The pipeline can be installed as a Python package via `pip`. It is recommended
to use a dedicated Python environment (e.g., Conda).

```bash
# Create a Conda environment (optional)
conda create -n rnacompete python=3.8 -y
conda activate rnacompete

# Install MEME Suite (required for motif operations)
conda install -c bioconda meme==5.5.5 -y

# Install rnacompete from the local directory
pip install .
```
All Python dependencies are installed automatically, including:
- logomaker
- matplotlib
- numpy
- scikit-learn
- scipy
- pandas
- pickleshare

After installation, the `rnacompete` command will be available:
```bash
rnacompete --help
```

## Input Data
### File structure
RNAcompete data is processed **in batches**. Input must follow this structure:
```
root
├── HybID00025_00103
│   ├── probe_intensity.tsv
├── HybID00110_00146
│   ├── probe_intensity.tsv
│── HybIDAAAAA_BBBBB
│   ├── probe_intensity.tsv
├── rnacompete_metadata.tsv
```

- `root/`: The folder containing all the data. Can take any name.
- `HybIDAAAAA_BBBBB/`: The folder containing data from a given batch. Can take
any name.

### Required Files
- `probe_intensity.tsv`: Probe intensities and flags for one batch.
  - Must have unique probe IDs (rows), a subset of
[`rnacompete/data/probe_metadata.tsv`](https://github.com/morrislab/rnacompete/blob/main/rnacompete/data/probe_metadata.tsv).
  - Each experiment is represented by **two consecutive columns**:
    - `HybIDXXXXX`: Float probe intensities (0-65535).
    - `HybIDXXXXX_flag`: Integer flags. Non-zero values indicate problematic
probes.

Example:

|                   | HybID00025 | HybID00025_flag | ... | HybIDXXXXX | HybIDXXXXX_flag |
|-------------------|------------|-----------------|-----|------------|-----------------|
| **RBD_v3_000001** | 368.5      | 0               | ... | 210.0      | 0               |
| **RBD_v3_000002** | 238.0      | 0               | ... | 53.0       | 1               |
| **...**           | ...        | ...             | ... | ...        | ...             |
| **RBD_v3_241399** | 284.5      | 0               | ... | 135.5      | 0               |
- `rnacompete_metadata.tsv`: Experimental metadata across all batches.
  - Must have unique experimental IDs (rows).
  - Must contain all experiments with intensities.
  - Required columns:
    - `rnacompete_id`: Secondary ID (optional).
    - `tax_name`: Source species (optional).
    - `gene_name`: Gene name (optional).
    - `normalization_batch`: Batch folder name.

Example:

| hyb_id         | rnacompete_id | tax_name                 | gene_name | normalization_batch |
|----------------|---------------|--------------------------|-----------|---------------------|
| **HybID00025** | RNCMPT00112   | Homo sapiens             | ELAVL1    | HybID00025_00103    |
| **HybID00026** |               | Drosophila melanogaster  | mub       | HybID00025_00103    |
| **...**        | ...           | ...                      | ...       | ...                 |
| **HybIDXXXXX** | RNCMPTYYYYY   |                          |           | HybIDAAAAA_BBBBB    |

## Usage
The pipeline can be run via a **Command-Line Interface (CLI)** or a Python
**API**.

It supports two modes:
1. **Processing** individual batches
2. **Summarization** across one or more batches.

Typically, different batches are first processed, whose outputs are then
summarized.

Example test data are provided in [`test`](https://github.com/morrislab/rnacompete/blob/main/test).

### 1. Processing
Process raw probe intensities into normalized Z-scores, motifs, sequence logos,
and classification results.

Parallelized across cores via `--n-jobs`(**CLI**) or `n_jobs`(**API**). 
#### CLI
```bash
rnacompete process \
--root test/root \
--batch HybID00025_00103_subset \
--n-jobs 8
```

#### API
```python
from rnacompete import run_process

res_dict = run_process(
    root='test/root',
    batch='HybID00025_00103_subset',
    n_jobs=8
)
```

#### Outputs
Results are saved to the `batch` folder and/or returned as the dictionary
`res_dict` (**API**).

| Output                    | CLI file/folder       | API key              |
|---------------------------|-----------------------|----------------------|
| Probe intensities (input) | `probe_intensity.tsv` | `probe_intensity_df` |
| Probe Z-scores            | `probe_zscore.tsv`    | `probe_zscore_df`    |
| 7-mer Z-scores            | `kmer_zscore.tsv`     | `kmer_zscore_df`     |
| Scatter plots             | `scatter/`            | –                    |
| PWMs                      | `pwm/`                | `pwm_list`           |
| Sequence logos            | `logo/`               | –                    |
| Classifier features       | `feature.tsv`         | `feature_df`         |
| Output summary            | `summary.tsv`         | `summary_df`         |

> **Notes:**
> - In the **API**, outputs are not written to disk if `save_results=False`.
> - In the **API**, PWMs in `pwm_list` match the column order of
> `kmer_zscore_df`.

### 2. Summarization
Aggregates experiments from one or more processed batches into combined summary
tables and, optionally, interactive HTML reports.

You must specify one of the following options:

| Option                                   | CLI                                  | API                                  |
|------------------------------------------|--------------------------------------|--------------------------------------|
| Summarize all experiments across batches | `--summarize-all`                    | `summarize_all=True`                 |
| Summarize all experiments in one batch   | `--summarize-batch HybIDAAAAA_BBBBB` | `summarize_batch='HybIDAAAAA_BBBBB'` |
| Summarize experiments across batches     | `--summarize-experiments exp.txt`    | `summarize_experiments='exp.txt'`    |

> **Notes:**
> - `exp.txt` (can take any name) contains one experiment ID per line.

#### CLI
```bash
rnacompete summarize \
--root test/root \
--output test/summarized_output \
--summarize-all
```

#### API
```python
from rnacompete import run_summarize

res_dict = run_summarize(
    root='test/root',
    output_path='test/summarize_output',
    summarize_all=True
)
```

#### Outputs
Results are saved in the `--output`(**CLI**)/`output_path`(**API**) directory
and/or returned as the dictionary `res_dict` (**API**).

| Output              | CLI file/folder | API key       |
|---------------------|-----------------|---------------|
| Classifier features | `feature.tsv`   | `feature_df`  |
| Output summary      | `summary.tsv`   | `summary_df`  |
| HTML reports        | `html/`         | –             |

> **Notes:**
> - HTML reports are skipped if `--no-html` (**CLI**).
> - In the **API**, outputs are not written to disk if `save_results=False`.

## License
This project is licensed under the BSD 3-Clause License. See
[LICENSE](https://github.com/morrislab/rnacompete/blob/main/LICENSE) for
details.

## References
If you use RNAcompete data in publications, please cite one of the following:

Ray, D., Kazan, H., Cook, K. et al.
**A compendium of RNA-binding motifs for decoding gene regulation.**
*Nature* (2013). https://doi.org/10.1038/nature12311

Ray, D., Laverty, K.U., Jolma, A. et al.
**RNA-binding proteins that lack canonical RNA-binding domains are rarely 
sequence-specific.**
*Sci Rep* (2023). https://doi.org/10.1038/s41598-023-32245-9

Sasse, A., Ray, D., Laverty, K.U. et al.
**A resource of RNA-binding protein motifs across eukaryotes reveals
evolutionary dynamics and gene-regulatory function.**
*Nat Biotechnol* (2025). https://doi.org/10.1038/s41587-025-02733-6
