from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingParameters:
    """Parameters for training the model."""

    # Training parameters
    learning_rate: float
    batch_size: int
    epochs: int

    # Model parameters
    model_name: str
    pretrained: str
    freeze_vision: bool = True
    precision: str = "bf16-mixed"

    # Dataset parameters
    data_dir: str = "./data"
    csv_filename: str = "ccneg_paraphrased.csv"
    image_dir_name: str = "ccneg_images/cc3m_subset_images_extracted_final"
    dataset_type: str = "ccneg"
    max_items: Optional[int] = None

    # Execution parameters
    num_workers: int = 4
    random_seed: int = 42
    device: str = "cuda"

    # Optimizer parameters
    weight_decay: float = 0.2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Loss function parameters
    paraphrase_weight: float = 1.0
    negation_weight: float = 1.0
    normalize_projections: bool = True
    num_projection_vectors: int = 2
    use_learnable_projections: bool = False
    temperature: float = 0.07

    # Training loop parameters
    gradient_accumulation_steps: int = 2
    gradient_clipping: Optional[float] = 1.0

    # Scheduler parameters
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    warmup_steps: int = 50

    # Cosine annealing parameters
    cosine_eta_min: float = 1e-6
    cosine_t_mult: int = 1

    # MLflow parameters
    experiment_name: str = "semclip"
    skip_final_checkpoint: bool = True
    skip_checkpoints: bool = True

    @classmethod
    def from_args(cls, args):
        """Create TrainingParameters from argparse arguments."""
        return cls(
            # Model parameters
            model_name=args.model_name,
            pretrained=args.pretrained,
            freeze_vision=args.freeze_vision,
            precision=args.precision,
            # Dataset parameters
            data_dir=args.data_dir,
            csv_filename=args.csv_filename,
            image_dir_name=args.image_dir_name,
            dataset_type=args.dataset_type,
            max_items=args.max_items,
            # Execution parameters
            device=args.device,
            # Optimizer parameters
            learning_rate=args.lr,
            # Loss parameters
            paraphrase_weight=args.paraphrase_weight,
            negation_weight=args.negation_weight,
            normalize_projections=args.normalize_projections,
            num_projection_vectors=args.num_projection_vectors,
            use_learnable_projections=args.use_learnable_projections,
            # Training loop parameters
            batch_size=args.batch_size,
            epochs=args.epochs,
            # MLflow parameters
            experiment_name=args.experiment_name,
            skip_final_checkpoint=args.skip_final_checkpoint,
            skip_checkpoints=args.skip_checkpoints,
        )
