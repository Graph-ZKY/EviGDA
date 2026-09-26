# EviGDA

Training and evaluation code for seven EviGDA graph domain adaptation targets: **CSBM-G4**, **DBLPv7**, **ACMv9**, **Citationv1**, **PROTEINS_P4**, **Mutagenicity_M4**, and **FRANKENSTEIN_F4**.

The repository includes frozen task configurations, processed target data, 18 pretrained source checkpoints, and seven adapted checkpoints. Use [main.py](main.py) to verify the files, evaluate the adapted models, or repeat adaptation from the source checkpoints.

## Installation

Download the repository and open a terminal in its root directory. Create a Conda environment and install the reference dependencies:

```bash
conda create -n evigda python=3.7.16 pip=23.1.2 -y
conda activate evigda
python -m pip install setuptools==68.0.0 wheel==0.42.0
python -m pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install numpy==1.21.6 scipy==1.7.3
python -m pip install torch-scatter==2.1.1+pt113cu117 torch-sparse==0.6.17+pt113cu117 --only-binary=torch-scatter,torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
python -m pip install -r requirements.txt
```

The reference environment uses Python 3.7.16, PyTorch 1.13.1 with CUDA 11.7, and PyTorch Geometric 2.3.1. Recorded training runs used an NVIDIA RTX 3080 with 10 GB of memory and four CPU threads. Checkpoint evaluation also supports CPU execution.

## Usage

Run all commands from the repository root.

### Verify repository files

```bash
python -B main.py --mode check
```

This checks the size and SHA-256 hash of every file listed in [MANIFEST.json](MANIFEST.json). All execution modes perform this check before continuing.

### Evaluate the supplied checkpoints

```bash
python -B main.py --mode evaluate --device cpu --output outputs/evaluate_table1
```

Evaluation loads the seven adapted checkpoints and runs a model forward pass on the target data. Stored predictions serve as reproducibility references. To evaluate on a GPU, use `--device cuda:0`.

### Train from the source checkpoints

```bash
python -B main.py --mode train --device cuda:0 --output outputs/train_table1
```

Training runs adaptation once for each target using the supplied source checkpoints and the settings in [configs/table1.json](configs/table1.json). It then reloads the resulting models to evaluate their predictions.

### Run selected targets

Pass exact target names to `--targets`, for example:

```bash
python -B main.py --mode evaluate --device cpu --targets DBLPv7 ACMv9 Citationv1 --output outputs/evaluate_citation
```

Omitting `--targets` runs all seven tasks. Use a **new output directory** for every training or evaluation run; existing directories are never overwritten. Relative output paths are resolved from the repository root. Run `python main.py --help` for the available arguments.

## Repository structure

| Path | Description |
| --- | --- |
| [main.py](main.py) | Entry point for file verification, training, and evaluation |
| [evigda/](evigda/) | Data loading, models, losses, memory, parameter averaging, and training |
| [configs/table1.json](configs/table1.json) | Frozen configurations for all seven targets |
| [data/](data/) | Processed target datasets |
| [mypretrain/](mypretrain/) | The 18 source checkpoints used for adaptation |
| [checkpoints/](checkpoints/) | Seven adapted models and reference predictions |
| [results/reference_results.json](results/reference_results.json) | Recorded correct counts and accuracies for reproduction checks |
| [MANIFEST.json](MANIFEST.json) | File sizes and SHA-256 hashes for integrity checks |
| [requirements.txt](requirements.txt) | Pinned Python dependencies |

The `outputs/` directory is created when training or evaluation runs. All required target data and model checkpoints are included in the repository.
