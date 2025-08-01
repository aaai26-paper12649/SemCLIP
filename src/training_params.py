from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingParameters:
    """Parameters for training the model."""

    # Required parameters
    learning_rate: float
    batch_size: int
    epochs: int

    # Model parameters
    freeze_vision: bool = True

    # Optimizer parameters
    weight_decay: float = 0.2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Loss function parameters
    temperature: float = 0.07
    paraphrase_weight: float = 1.0
    negation_weight: float = 1.0
    normalize_projections: bool = True
    num_projection_vectors: int = 2
    use_learnable_projections: bool = False

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

    @classmethod
    def from_args(cls, args):
        """Create TrainingParameters from argparse arguments."""
        return cls(
            # Model parameters
            freeze_vision=args.freeze_vision,
            # Optimizer parameters
            learning_rate=args.lr,
            # Loss parameters
            temperature=args.temperature,
            paraphrase_weight=args.paraphrase_weight,
            negation_weight=args.negation_weight,
            normalize_projections=args.normalize_projections,
            num_projection_vectors=args.num_projection_vectors,
            use_learnable_projections=args.use_learnable_projections,
            # Training loop parameters
            batch_size=args.batch_size,
            epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
