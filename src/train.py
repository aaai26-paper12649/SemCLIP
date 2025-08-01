import os
import torch
import lightning as L
import datetime
import shutil
import mlflow
import tempfile
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import MLFlowLogger
from args import parse_args
from training_params import TrainingParameters
from model import SemCLIP


def set_random_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_random_seed()


def setup_callbacks(args, logger=None):
    """Configure and return Lightning callbacks based on training arguments.

    Args:
        args: Command-line arguments containing callback configuration options
        logger: MLFlow logger for retrieving artifact paths

    Returns:
        tuple: (list of callbacks, best model checkpoint callback or None)
    """
    callbacks = [LearningRateMonitor(logging_interval="step")]

    best_model_callback = None

    class TempModelCheckpoint(ModelCheckpoint):
        def __init__(self, *args, **kwargs):
            self.temp_dir = tempfile.mkdtemp(prefix="checkpoints_")
            if "dirpath" in kwargs:
                kwargs["dirpath"] = self.temp_dir
            else:
                kwargs["dirpath"] = self.temp_dir

            if "save_top_k" not in kwargs:
                kwargs["save_top_k"] = 1

            super().__init__(*args, **kwargs)

    if not args.max_items:
        if not args.skip_checkpoints and logger is not None:

            class MLflowModelCheckpoint(TempModelCheckpoint):
                def __init__(self, *args, **kwargs):
                    self.mlflow_run_id = logger.run_id
                    self.mlflow_experiment_id = logger.experiment_id
                    super().__init__(*args, **kwargs)

            best_model_callback = MLflowModelCheckpoint(
                filename="semclip-best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
                verbose=True,
            )
        else:
            best_model_callback = TempModelCheckpoint(
                filename="semclip-best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=False,
                verbose=True,
            )

        callbacks.append(best_model_callback)

    return [cb for cb in callbacks if cb is not None], best_model_callback


def setup_mlflow_logging(args, training_params):
    """Configure MLflow logging and log all hyperparameters.

    Args:
        args: Command-line arguments
        training_params: Training parameters object

    Returns:
        MLFlowLogger: Configured logger instance
    """
    experiment_name = "semclip"

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{timestamp}"

    logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
    )

    model_type = "baseline"
    if training_params.paraphrase_weight == 0 and training_params.negation_weight == 0:
        model_type = "baseline"
    elif (
        training_params.paraphrase_weight == 1 and training_params.negation_weight == 0
    ):
        model_type = "paraphrase_only"
    elif (
        training_params.paraphrase_weight == 0 and training_params.negation_weight == 1
    ):
        model_type = "negation_only"
    elif (
        training_params.paraphrase_weight == 1 and training_params.negation_weight == 1
    ):
        model_type = "ours"

    hyperparams = {
        # Model configuration
        "model": args.model,
        "pretrained": args.pretrained,
        "freeze_vision": args.freeze_vision,
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
        "data_dir": args.data_dir,
        "csv_filename": args.csv_filename,
        "image_dir_name": args.image_dir_name,
        "max_items": args.max_items,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "random_seed": args.random_seed,
        "num_workers": args.num_workers,
        # Advanced training options
        "gradient_accumulation_steps": training_params.gradient_accumulation_steps,
        "gradient_clipping": training_params.gradient_clipping,
        "use_scheduler": training_params.use_scheduler,
        # Hardware and storage
        "device": args.device,
        "precision": args.precision,
        "skip_final_checkpoint": args.skip_final_checkpoint,
        "skip_checkpoints": args.skip_checkpoints,
    }

    hyperparams = {k: v for k, v in hyperparams.items() if v is not None}

    logger.log_hyperparams(hyperparams)

    return logger


def cleanup_temp_files():
    """Clean up temporary files created during training."""
    import subprocess
    import re
    import glob

    temp_ckpt_patterns = [".scale_batch_size_*.ckpt", ".lr_find_*.ckpt"]

    for pattern in temp_ckpt_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"Removed temporary file: {file_path}")
            except Exception as e:
                print(f"Could not remove file {file_path}: {e}")

    try:
        print("Running MLflow garbage collection...")
        subprocess.run(["mlflow", "gc"], check=True, capture_output=True, text=True)
        print("MLflow garbage collection completed successfully.")
    except Exception as e:
        print(f"MLflow garbage collection failed: {e}")

    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    if os.path.exists(mlflow_tracking_uri):
        for item in os.listdir(os.getcwd()):
            item_path = os.path.join(os.getcwd(), item)

            if os.path.isdir(item_path) and (
                item.isdigit() or re.match(r"^\d{18,19}$", item)
            ):
                try:
                    shutil.rmtree(item_path)
                    print(f"Removed orphaned experiment directory: {item_path}")
                except Exception as e:
                    print(f"Could not remove directory {item_path}: {e}")


def restore_best_checkpoint(args, model, best_model_callback, logger):
    """Restore the best model checkpoint and optionally save to MLflow.

    Args:
        args: Command-line arguments
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

        if not args.skip_checkpoints and logger is not None:
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
    checkpoint_name = f"finetuned_open_clip_sem_{timestamp}.ckpt"

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

    set_random_seed(args.random_seed)

    model = SemCLIP(
        model_name=args.model,
        pretrained=args.pretrained,
        training_params=training_params,
        data_dir=args.data_dir,
        csv_filename=args.csv_filename,
        image_dir_name=args.image_dir_name,
        image_extension=args.image_extension,
        batch_size=training_params.batch_size,
        max_items=args.max_items,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
    )

    logger = setup_mlflow_logging(args, training_params)

    callbacks, best_model_callback = setup_callbacks(args, logger)

    trainer = L.Trainer(
        max_epochs=training_params.epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=training_params.gradient_accumulation_steps,
        gradient_clip_val=(
            training_params.gradient_clipping
            if training_params.gradient_clipping
            else 1.0
        ),
        log_every_n_steps=10,
        check_val_every_n_epoch=3,
    )

    if not args.test_only:
        trainer.fit(model)
        model = restore_best_checkpoint(args, model, best_model_callback, logger)

        if not args.skip_final_checkpoint:
            save_final_checkpoint_to_mlflow(trainer, logger)

        print("Training completed successfully!")
    else:
        print("Skipping training phase, running test only...")

    trainer.test(model)

    cleanup_temp_files()


if __name__ == "__main__":
    main()
