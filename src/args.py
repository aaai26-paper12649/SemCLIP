import argparse

import torch


def parse_args():
    """
    Parse command-line arguments for fine-tuning open_clip with custom SemCLIP loss.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune open_clip with custom SemCLIP loss"
    )
    # Model parameters
    parser.add_argument(
        "--model-name", type=str, default="ViT-B-32", help="open_clip model name"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="open_clip pretrained tag",
    )
    parser.add_argument(
        "--paraphrase-weight",
        type=float,
        default=1.0,
        help="Weight for paraphrase loss",
    )
    parser.add_argument(
        "--negation-weight", type=float, default=1.0, help="Weight for negation loss"
    )
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        default=True,
        help="Freeze the vision encoder and only train the text encoder (default: True)",
    )
    parser.add_argument(
        "--no-freeze-vision",
        action="store_false",
        dest="freeze_vision",
        help="Don't freeze the vision encoder, train both vision and text encoders",
    )
    parser.add_argument(
        "--normalize-projections",
        action="store_true",
        default=True,
        help="Normalize projections to unit length before computing similarity",
    )
    parser.add_argument(
        "--no-normalize-projections",
        action="store_false",
        dest="normalize_projections",
        help="Do not normalize projections (use raw dot products)",
    )
    parser.add_argument(
        "--num-projection-vectors",
        type=int,
        default=2,
        help="Number of basis vectors for projection (default: 2)",
    )
    parser.add_argument(
        "--use-learnable-projections",
        action="store_true",
        default=False,
        help="Use learnable projection vectors instead of fixed ones",
    )
    parser.add_argument(
        "--no-learnable-projections",
        action="store_false",
        dest="use_learnable_projections",
        help="Use fixed projection vectors",
    )

    # Dataset parameters
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Path to the data directory"
    )
    parser.add_argument(
        "--csv-filename",
        type=str,
        default="ccneg_paraphrased.csv",
        help="Name of the CSV file within data_dir (default: ccneg_paraphrased.csv)",
    )
    parser.add_argument(
        "--image-dir-name",
        type=str,
        default="ccneg_images/cc3m_subset_images_extracted_final",
        help="Name of the image subdirectory within data_dir (default: ccneg_images/cc3m_subset_images_extracted_final)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="ccneg",
        choices=["ccneg", "sugarcrepe"],
        help="Type of dataset to use (default: ccneg)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to load from the dataset (default: None, load all)",
    )

    # Training parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training (default: 256)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Initial learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed",
        choices=["16-mixed", "32-true", "bf16-mixed"],
        help="Precision for training (default: 16-mixed)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data loading"
    )

    # Hardware parameters
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # MLflow parameters
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="semclip",
        help="MLflow experiment name",
    )

    # Disk space management
    parser.add_argument(
        "--skip-final-checkpoint",
        action="store_true",
        default=True,
        help="Skip saving the final model checkpoint to save disk space (default: True)",
    )
    parser.add_argument(
        "--save-final-checkpoint",
        action="store_false",
        dest="skip_final_checkpoint",
        help="Save the final model checkpoint",
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        default=True,
        help="Skip saving any checkpoints during training, including the best model (default: True)",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_false",
        dest="skip_checkpoints",
        help="Save checkpoints during training",
    )

    return parser.parse_args()
