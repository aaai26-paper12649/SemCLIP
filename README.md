# SemCLIP

Fine-tune OpenCLIP with a custom loss that leverages paraphrased and negated captions to create a more robust semantic alignment between text and image.

## Project Structure

### Source Code (`/src`)

- `__init__.py`: Package initialization file
- `custom_loss.py`: Implements the NegationParaphraseProjectionLoss combining CLIP, paraphrasing, and negation objectives.
- `dataset.py`: Dataset and collate function for images with original, paraphrased, and negated captions.
- `model.py`: PyTorch Lightning implementation of the SemCLIP model.
- `train.py`: Training script using PyTorch Lightning.
- `config.py`: Configuration classes for model architecture.
- `args.py`: Command-line argument parsing.
- `training_params.py`: Training parameters and hyperparameters.
- `evaluation.py`: Evaluation metrics and utilities.

## Dataset Requirements

To train with this project, you must manually add the required datasets to the `data/` directory:

- **CCNeg** (for `--dataset-type ccneg`):
  - Download and prepare the CCNeg dataset by following the instructions here: [CCNeg Dataset Instructions](https://github.com/jaisidhsingh/CoN-CLIP/blob/main/ccneg_dataset/README.md)

- **SugarCrepe++** (for `--dataset-type sugarcrepe`):
  - Download and prepare the SugarCrepe++ dataset: [SugarCrepe++ Dataset (arXiv)](https://arxiv.org/abs/2406.11171)

See below for the expected CSV format and directory structure for each dataset type.

## Dataset Format

The dataset is loaded from a CSV file with the following columns:

### Image Placement and Naming

- **For `ccneg` dataset type:**
  - Images must be named as 9-digit zero-padded numbers based on the `image_number` column in the CSV, with the appropriate extension (default `.jpg`).
    - Example: `image_number` 2 → `000000002.jpg`
  - Place all images in the directory specified by `--image-dir-name` (default: `images`), inside your `--data-dir`.

- **For `sugarcrepe` dataset type:**
  - Images must be named exactly as specified in the `filename` column of the CSV (case-sensitive).
  - Place all images in the directory specified by `--image-dir-name` (default: `images`), inside your `--data-dir`.

Example directory structure for both:
```
data/
  ccneg_paraphrased.csv
  sugarcrepe.csv
  images/
    000000000.jpg
    000000001.jpg
    image1.jpg
    image2.jpg
    ...
```


- `image_number`: A numeric identifier for the image (e.g., `2`). Will be zero-padded to 9 digits for filename construction (e.g., `000000002.jpg`).
- `caption`: The original caption (raw string, **not tokenized**).
- `negation`: The negated caption (raw string, **not tokenized**).
- `paraphrased`: The paraphrased caption (raw string, **not tokenized**).


**Note:** Captions from the CSV are tokenized by the `SemCLIPDataset` and `custom_collate_fn` before being passed to the model.

### Caption Generation (`/syn_caption_generation`)

- `syn_caption_generation.ipynb`: Jupyter notebook for generating paraphrased and negated captions using Large Language Models (LLMs). This notebook provides an utility to automatically create synthetic caption variations from original image captions, which are essential for training the SemCLIP model with paraphrase and negation captions.

The processed data for CCNeg and Sugarcrepe++ have been provided in the \data folder as ccneg_paraphrased.csv and sugarcrepe_pp_paraphrased.csv.

## Methodology

SemCLIP extends CLIP by engineering a new loss function to incorporate the concepts of paraphrasing and negation. The objective is to enrich the joint embedding space with these concepts, producing a more robust semantic alignment between text and image.

### Loss Function

The model uses a custom `NegationParaphraseProjectionLoss` that combines:
1. Standard CLIP contrastive loss (`L_contrastive`)
2. Paraphrase consistency loss (`L_paraphrase = 1 - cos(p(t), p(t+))`) 
3. Negation discrimination loss (`L_negation = max(0, cos(p(t), p(t-)))`) 

The total loss is a weighted combination:
```
L_total = (α*L_contrastive + β*L_paraphrase + γ*L_negation) / (α + β + γ)
```

Where α, β, and γ are weights for each loss component.

### Embedding Projections

The model uses a projection mechanism to create a lower-dimensional subspace where semantic relationships are enforced. Key parameters:
- `num_projection_vectors`: Number of projection directions (1 or 2)
- `normalize_projections`: Whether to normalize the projections to unit length
- `use_learnable_projections`: Whether projection vectors can be updated during training

## Logging

Metric logging is handled by PyTorch Lightning's `self.log` method. Experiment tracking can be managed using MLflow, as detailed in the "MLflow Tracking" section below.

## Installation and Usage

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. To set up the project with uv:

```bash
# Install uv if you don't have it already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the project and its dependencies
# Install required dependencies
uv pip install -e ".[dev]"  # Includes development dependencies like pytest and coverage
```

### Alternative Installation

If you prefer using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Training

To train the model with default settings:

```bash
uv run python src/train.py --batch-size 64 --epochs 10
```


### Dataset Types

SemCLIP supports multiple dataset formats through the `--dataset-type` parameter:

- `ccneg` (default): Uses the CCNeg dataset format where images are referenced by `image_number`
- `sugarcrepe`: Uses the SugarCrepe++ dataset format where images are referenced by `filename`

Example:

```bash
uv run python src/train.py --dataset-type sugarcrepe
```

Change model architecture and pretrained weights (unfreeze vision encoder):

```bash
uv run python src/train.py --model-name ViT-B-32 --pretrained laion2b_s34b_b79k --no-freeze-vision
```

Train with specific loss weights (e.g., only contrastive and paraphrase losses, no negation):

```bash
uv run python src/train.py --paraphrase-weight 1.0 --negation-weight 0.0
```

### Key Training Parameters

- `--freeze-vision/--no-freeze-vision`: Control whether to freeze the vision encoder (default: frozen)
- `--model`: Model architecture (default: ViT-B-32)
- `--pretrained`: Pretrained weights (default: laion2b_s34b_b79k)
- `--batch-size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (default: 5e-5)

#### Loss Function Parameters

- `--paraphrase-weight`: Weight for paraphrase loss (default: 1.0)
- `--negation-weight`: Weight for negation loss (default: 1.0)
- `--num-projection-vectors`: Number of projection vectors (default: 2)
- `--normalize-projections`: Whether to normalize projections (default: True)
- `--use-learnable-projections`: Whether projection vectors are learnable (default: False)

The training script includes optimized default parameters and advanced features:

- Mixed precision training (16-bit)
- Cosine annealing scheduler

**Note:** Only parameters listed in the CLI are user-configurable. Other advanced parameters use defaults as defined in the codebase (`training_params.py`).

### Automatic Optimization Features

Additional dataset parameters (with defaults):

```bash
--data-dir ./data  # Path to the data directory
--csv-filename dataset.csv  # Name of the CSV file
--image-dir-name images  # Path to images relative to data_dir
--dataset-type ccneg  # Type of dataset to use (default: ccneg)
--random-seed 42  # Random seed for reproducible dataset splits
```

## Notes

- Input caption files should contain raw strings. Tokenization and padding are handled by the `SemCLIPDataset` and `custom_collate_fn` as part of the data loading pipeline.

## Evaluation Metrics

The model evaluation uses several metrics:

- **Top-k accuracy**: Measures how often the correct caption is in the top k predictions for an image.
- **Original vs Negation accuracy**: Measures the model's ability to distinguish between an original caption and its negation.
- **Scaled Original vs Negation accuracy**: Adjusts the original vs negation score to account for the 50% random baseline.
- **Combined score**: The average of Top-1 original accuracy, Top-1 paraphrase accuracy, and the scaled original vs negation score.

## MLflow Tracking

This project uses MLflow for experiment tracking. To view the experiment results:

```bash
# Start the MLflow UI server
uv run mlflow ui

```

Then open your browser to <http://localhost:5000>

## Evaluation

### Automatic Evaluation

Zero-shot evaluation on standard image classification datasets is automatically performed at the end of training as part of the PyTorch Lightning test phase. This includes:

1. Evaluation on multiple datasets (CIFAR10, CIFAR100, Caltech101, etc.)
2. Testing with both original and negated prompts
3. Logging of results to MLflow
