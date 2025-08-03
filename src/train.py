import datetime
import os
import tempfile

import lightning as L
import mlflow
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger

from args import parse_args
from model import SemCLIP
from training_params import TrainingParameters


def set_random_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_random_seed()


def setup_callbacks(training_params, logger=None):
    """Configure and return Lightning callbacks based on training arguments.

    Args:
        training_params: Training parameters object
        logger: MLFlow logger for retrieving artifact paths (unused in simplified version)

    Returns:
        tuple: (list of callbacks, best model checkpoint callback or None)
    """

    callbacks = [LearningRateMonitor(logging_interval="step")]
    best_model_callback = None

    if not getattr(training_params, "max_items", False) and not getattr(training_params, "skip_checkpoints", False):
        best_model_callback = ModelCheckpoint(
            filename="semclip-best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            verbose=True,
        )
        callbacks.append(best_model_callback)

    return callbacks, best_model_callback


def setup_mlflow_logging(training_params):
    """Configure MLflow logging and log all hyperparameters.

    Args:
        training_params: Training parameters object

    Returns:
        MLFlowLogger: Configured logger instance
    """
    experiment_name = training_params.experiment_name

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{training_params.model_name}_{timestamp}"

    logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
    )

    weights = (training_params.paraphrase_weight, training_params.negation_weight)
    match weights:
        case (0.0, 0.0):
            model_type = "baseline"
        case (1.0, 0.0):
            model_type = "paraphrase_only"
        case (0.0, 1.0):
            model_type = "negation_only"
        case (1.0, 1.0):
            model_type = "semclip"
        case _:
            model_type = "custom"

    hyperparams = {
        # Model configuration
        "model_name": training_params.model_name,
        "pretrained": training_params.pretrained,
        "freeze_vision": training_params.freeze_vision,
        "model_type": model_type,
        # Training parameters
        "learning_rate": training_params.learning_rate,
        "weight_decay": training_params.weight_decay,
        "batch_size": training_params.batch_size,
        "epochs": training_params.epochs,
        "temperature": training_params.temperature,
        # Loss function weights and parameters
        "paraphrase_weight": training_params.paraphrase_weight,
        "negation_weight": training_params.negation_weight,
        "normalize_projections": training_params.normalize_projections,
        "num_projection_vectors": training_params.num_projection_vectors,
        "use_learnable_projections": training_params.use_learnable_projections,
        # Dataset parameters
        "data_dir": training_params.data_dir,
        "csv_filename": training_params.csv_filename,
        "image_dir_name": training_params.image_dir_name,
        "max_items": training_params.max_items,
        "random_seed": training_params.random_seed,
        "num_workers": training_params.num_workers,
        # Advanced training options
        "gradient_accumulation_steps": training_params.gradient_accumulation_steps,
        "gradient_clipping": training_params.gradient_clipping,
        "use_scheduler": training_params.use_scheduler,
        # Hardware and storage
        "device": training_params.device,
        "precision": training_params.precision,
        "skip_final_checkpoint": training_params.skip_final_checkpoint,
        "skip_checkpoints": training_params.skip_checkpoints,
    }

    hyperparams = {k: v for k, v in hyperparams.items() if v is not None}

    logger.log_hyperparams(hyperparams)

    return logger

def restore_best_checkpoint(training_params, model, best_model_callback, logger):
    """Restore the best model checkpoint and optionally save to MLflow.

    Args:
        training_params: Training parameters object
        model: The current model
        best_model_callback: The best model checkpoint callback
        logger: MLflow logger

    Returns:
        The model with restored weights (either best checkpoint or current state)
    """
    if (
        best_model_callback is not None
        and best_model_callback.best_model_path
        and os.path.exists(best_model_callback.best_model_path)
    ):
        print(
            f"\nLoading best model from checkpoint: {best_model_callback.best_model_path}"
        )
        checkpoint = torch.load(best_model_callback.best_model_path)
        model.load_state_dict(checkpoint["state_dict"])
        print(
            f"Loaded best model with validation loss: {best_model_callback.best_model_score:.4f} (lower is better)"
        )

        if not training_params.skip_checkpoints and logger is not None:
            with mlflow.start_run(run_id=logger.run_id):
                mlflow.log_artifact(
                    best_model_callback.best_model_path,
                    artifact_path="checkpoints/best_model",
                )
    else:
        print(
            "\nWARNING: No best model checkpoint available. Using final model state for testing."
        )

    return model


def save_final_checkpoint_to_mlflow(trainer, logger):
    """Save the final model checkpoint to MLflow artifacts.

    Args:
        trainer: Lightning Trainer instance with trained model
        logger: MLflow logger for logging the checkpoint
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"finetuned_open_clip_semclip_{timestamp}.ckpt"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_checkpoint_path = os.path.join(temp_dir, checkpoint_name)
        trainer.save_checkpoint(temp_checkpoint_path)

        with mlflow.start_run(run_id=logger.run_id):
            mlflow.log_artifact(temp_checkpoint_path, artifact_path="model_checkpoints")

        print(
            f"Final model checkpoint saved to MLflow: model_checkpoints/{checkpoint_name}"
        )


def main():
    args = parse_args()
    training_params = TrainingParameters.from_args(args)

    set_random_seed(training_params.random_seed)

    model = SemCLIP(training_params=training_params)

    logger = setup_mlflow_logging(training_params)

    callbacks, best_model_callback = setup_callbacks(training_params, logger)

    trainer = L.Trainer(
        max_epochs=training_params.epochs,
        precision=training_params.precision,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=training_params.gradient_accumulation_steps,
        gradient_clip_val=training_params.gradient_clipping,
        log_every_n_steps=10,
        check_val_every_n_epoch=int(training_params.epochs / 10),
    )

    trainer.fit(model)

    model = restore_best_checkpoint(training_params, model, best_model_callback, logger)

    if not training_params.skip_final_checkpoint:
        save_final_checkpoint_to_mlflow(trainer, logger)

    print("Training completed successfully!. Running evaluation...")

    trainer.test(model)

    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
