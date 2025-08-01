import torch
import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from open_clip import create_model_and_transforms, get_tokenizer
from custom_loss import NegationParaphraseProjectionLoss
import os
from dataset import SemCLIPDataset, custom_collate_fn
from evaluation import (
    compute_topk_retrieval,
    compute_negation_retrieval,
    evaluate_zeroshot_on_all_datasets,
)


class SemCLIP(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        training_params,
        data_dir: Optional[str] = None,
        csv_filename: Optional[str] = None,
        image_dir_name: Optional[str] = None,
        dataset_type: str = "ccneg",
        max_items: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["training_params", "batch_size"])
        self.model_name = model_name
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / training_params.temperature))
        )
        self.pretrained = pretrained
        self.training_params = training_params
        self.data_dir = data_dir
        self.csv_filename = csv_filename
        self.image_dir_name = image_dir_name
        self.dataset_type = dataset_type
        self.max_items = max_items
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        self._init_model()

        self.loss_fn = NegationParaphraseProjectionLoss(
            paraphrase_weight=training_params.paraphrase_weight,
            negation_weight=training_params.negation_weight,
            normalize_projections=training_params.normalize_projections,
            num_projection_vectors=training_params.num_projection_vectors,
            use_learnable_projections=training_params.use_learnable_projections,
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _init_model(self):
        """Initialize the CLIP model"""
        self.model, _, self.preprocess = create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        if self.training_params.freeze_vision:
            self.model.lock_image_tower()
        self.tokenizer = get_tokenizer(self.model_name)

    def forward(self, images, captions=None):
        """Forward pass through the model."""
        image_emb = self.model.encode_image(images)

        if captions is not None:
            text_emb = self.model.encode_text(captions.to(images.device))
            return image_emb, text_emb
        return image_emb

    def training_step(self, batch, batch_idx):
        """Training step."""
        images = batch["image"]

        cap_tokens = batch["caption"].to(self.device)
        paraphrasing_tokens = batch["paraphrased_caption"].to(self.device)
        negation_tokens = batch["negated_caption"].to(self.device)

        image_emb = self.model.encode_image(images)
        text_emb = self.model.encode_text(cap_tokens)
        paraphrasing_emb = self.model.encode_text(paraphrasing_tokens)
        negation_emb = self.model.encode_text(negation_tokens)

        logit_scale = self.logit_scale.exp()
        contrastive_loss = self.loss_fn.clip_loss(image_emb, text_emb, logit_scale)
        loss = self.loss_fn(
            text_embeddings=text_emb,
            paraphrase_embeddings=paraphrasing_emb,
            negation_embeddings=negation_emb,
            contrastive_loss=contrastive_loss,
        )

        self.log("train_loss", loss["loss"], on_step=True, on_epoch=True, prog_bar=True)

        if "paraphrase_loss" in loss:
            self.log(
                "train_paraphrase_loss",
                loss["paraphrase_loss"],
                on_step=True,
                on_epoch=True,
            )
        if "negation_loss" in loss:
            self.log(
                "train_negation_loss",
                loss["negation_loss"],
                on_step=True,
                on_epoch=True,
            )
        if "contrastive_loss" in loss:
            self.log(
                "train_contrastive_loss",
                loss["contrastive_loss"],
                on_step=True,
                on_epoch=True,
            )

        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch["image"]

        cap_tokens = batch["caption"].to(self.device)
        paraphrasing_tokens = batch["paraphrased_caption"].to(self.device)
        negation_tokens = batch["negated_caption"].to(self.device)

        image_emb = self.model.encode_image(images)
        text_emb = self.model.encode_text(cap_tokens)
        paraphrasing_emb = self.model.encode_text(paraphrasing_tokens)
        negation_emb = self.model.encode_text(negation_tokens)

        logit_scale = self.logit_scale.exp()
        contrastive_loss = self.loss_fn.clip_loss(image_emb, text_emb, logit_scale)
        loss = self.loss_fn(
            text_embeddings=text_emb,
            paraphrase_embeddings=paraphrasing_emb,
            negation_embeddings=negation_emb,
            contrastive_loss=contrastive_loss,
        )

        result = {
            "val_loss": loss["loss"],
            "image_emb": image_emb.detach(),
            "text_emb": text_emb.detach(),
            "paraphrasing_emb": paraphrasing_emb.detach(),
            "negation_emb": negation_emb.detach(),
        }

        if "paraphrasing_loss" in loss:
            result["paraphrasing_loss"] = loss["paraphrasing_loss"]
        if "negation_loss" in loss:
            result["negation_loss"] = loss["negation_loss"]
        if "contrastive_loss" in loss:
            result["contrastive_loss"] = loss["contrastive_loss"]

        self.validation_step_outputs.append(result)

        self.log("val_loss", loss["loss"], on_step=False, on_epoch=True, prog_bar=True)

        if "paraphrasing_loss" in loss:
            self.log(
                "val_paraphrasing_loss",
                loss["paraphrasing_loss"],
                on_step=False,
                on_epoch=True,
            )
        if "negation_loss" in loss:
            self.log(
                "val_negation_loss", loss["negation_loss"], on_step=False, on_epoch=True
            )
        if "contrastive_loss" in loss:
            self.log(
                "val_contrastive_loss",
                loss["contrastive_loss"],
                on_step=False,
                on_epoch=True,
            )

        return result

    def _compute_and_log_metrics(self, outputs, prefix):
        """
        Compute retrieval metrics and log them using self.log.
        Args:
            outputs (list): List of dicts with keys 'image_emb', 'text_emb', 'paraphrasing_emb', 'negation_emb'
            prefix (str): 'val' or 'test'
        """
        if not outputs:
            return

        print(f"\nRunning zero-shot evaluation on all datasets for {prefix} phase...")
        if prefix == "test":
            try:
                all_metrics = evaluate_zeroshot_on_all_datasets(
                    model=self.model,
                    preprocess=self.preprocess,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    data_dir="./data",
                    batch_size=256,
                    num_workers=4,
                )

                total_standard_acc = 0.0
                total_negated_acc = 0.0
                num_datasets = len(all_metrics)

                for dataset_name, metrics in all_metrics.items():
                    total_standard_acc += metrics["standard_accuracy"]
                    total_negated_acc += metrics["negated_accuracy"]

                    for metric_name, value in metrics.items():
                        self.log(
                            f"{prefix}_{dataset_name}_zeroshot_{metric_name}",
                            value,
                            on_epoch=True,
                            prog_bar=True,
                            logger=True,
                            sync_dist=True,
                        )
                        print(
                            f"Logged {prefix}_{dataset_name}_zeroshot_{metric_name}: {value:.4f}"
                        )

                avg_standard = total_standard_acc / num_datasets
                avg_negated = total_negated_acc / num_datasets
                avg_delta = avg_standard - avg_negated

                for metric_name, value in [
                    ("standard_accuracy", avg_standard),
                    ("negated_accuracy", avg_negated),
                    ("delta", avg_delta),
                ]:
                    self.log(
                        f"{prefix}_avg_zeroshot_{metric_name}",
                        value,
                        on_epoch=True,
                        prog_bar=True,
                        logger=True,
                        sync_dist=True,
                    )
                    print(f"Logged {prefix}_avg_zeroshot_{metric_name}: {value:.4f}")

            except Exception as e:
                print(f"Error during zero-shot evaluation: {str(e)}")
                import traceback

                traceback.print_exc()

        print(f"Number of batches in outputs: {len(outputs)}")

        all_image_emb = torch.cat([x["image_emb"] for x in outputs])
        all_text_emb = torch.cat([x["text_emb"] for x in outputs])
        all_paraphrasing_emb = torch.cat([x["paraphrasing_emb"] for x in outputs])
        all_negation_emb = torch.cat([x["negation_emb"] for x in outputs])

        sample_size = min(100, all_text_emb.shape[0])
        sample_indices = torch.randperm(all_text_emb.shape[0])[:sample_size]
        text_sample = F.normalize(all_text_emb[sample_indices], dim=-1)
        paraphrasing_sample = F.normalize(all_paraphrasing_emb[sample_indices], dim=-1)
        negation_sample = F.normalize(all_negation_emb[sample_indices], dim=-1)

        text_paraphrasing_sim = torch.sum(text_sample * paraphrasing_sample, dim=1)
        text_negation_sim = torch.sum(text_sample * negation_sample, dim=1)

        print(
            f"  Text-Paraphrase: min={text_paraphrasing_sim.min().item():.4f}, max={text_paraphrasing_sim.max().item():.4f}, mean={text_paraphrasing_sim.mean().item():.4f}"
        )
        print(
            f"  Text-Negation: min={text_negation_sim.min().item():.4f}, max={text_negation_sim.max().item():.4f}, mean={text_negation_sim.mean().item():.4f}"
        )

        all_image_emb = F.normalize(all_image_emb, dim=-1)
        all_text_emb = F.normalize(all_text_emb, dim=-1)
        all_paraphrasing_emb = F.normalize(all_paraphrasing_emb, dim=-1)
        all_negation_emb = F.normalize(all_negation_emb, dim=-1)

        eval_batch_size = min(self.batch_size, 128)

        topk_original = compute_topk_retrieval(
            all_image_emb, all_text_emb, self.device, batch_size=eval_batch_size
        )

        topk_paraphrasing = compute_topk_retrieval(
            all_image_emb, all_paraphrasing_emb, self.device, batch_size=eval_batch_size
        )

        accuracy = compute_negation_retrieval(
            all_image_emb, all_text_emb, all_negation_emb, batch_size=eval_batch_size
        )

        for k, acc in topk_original.items():
            self.log(
                f"{prefix}_top{k}_original",
                acc,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        for k, acc in topk_paraphrasing.items():
            self.log(
                f"{prefix}_top{k}_paraphrased_retrieval_accuracy",
                acc,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        self.log(
            f"{prefix}_original_vs_negation_accuracy",
            accuracy,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        scaled_accuracy = max(0.0, (accuracy - 0.5) * 2.0)
        self.log(
            f"{prefix}_scaled_original_vs_negation_accuracy",
            scaled_accuracy,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        combined_score = (
            topk_original[1] + topk_paraphrasing[1] + scaled_accuracy
        ) / 3.0
        self.log(
            f"{prefix}_combined_score",
            combined_score,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_validation_epoch_end(self):
        """Compute validation metrics at the end of the validation epoch."""
        self._compute_and_log_metrics(self.validation_step_outputs, prefix="val")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step - same as validation step."""
        result = self.validation_step(batch, batch_idx)
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self):
        """Compute test metrics at the end of the test epoch."""
        self._compute_and_log_metrics(self.test_step_outputs, prefix="test")
        self.test_step_outputs.clear()

    @property
    def learning_rate(self):
        return self.training_params.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.training_params.learning_rate = value

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler with cosine schedule and warmup."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.training_params.weight_decay,
            betas=(self.training_params.beta1, self.training_params.beta2),
            eps=self.training_params.eps,
        )

        if hasattr(self.trainer, "estimated_stepping_batches"):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = len(self.train_dataloader()) * self.trainer.max_epochs

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.training_params.warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - self.training_params.warmup_steps,
            eta_min=0,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.training_params.warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        """Prepare training dataloader."""
        if not self.data_dir or not self.csv_filename or not self.image_dir_name:
            raise ValueError(
                "data_dir, csv_filename, and image_dir_name must be provided for training."
            )

        image_full_dir = os.path.join(self.data_dir, self.image_dir_name)

        train_dataset = SemCLIPDataset(
            data_dir=self.data_dir,
            csv_filename=self.csv_filename,
            image_dir=image_full_dir,
            tokenizer=self.tokenizer,
            split="train",
            max_items=self.max_items,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed,
            dataset_type=self.dataset_type,
        )
        num_workers = 4 if self.num_workers is None else self.num_workers
        dataloader_kwargs = {
            "dataset": train_dataset,
            "batch_size": self.batch_size,
            "shuffle": True,
            "collate_fn": custom_collate_fn,
            "num_workers": num_workers,
            "pin_memory": True,
        }

        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["prefetch_factor"] = 2

        return DataLoader(**dataloader_kwargs)

    def val_dataloader(self):
        """Prepare validation dataloader."""
        if not self.data_dir or not self.csv_filename or not self.image_dir_name:
            print(
                "Validation data parameters (data_dir, csv_filename, image_dir_name) not fully specified. Skipping validation."
            )
            return None

        image_full_dir = os.path.join(self.data_dir, self.image_dir_name)

        val_dataset = SemCLIPDataset(
            data_dir=self.data_dir,
            csv_filename=self.csv_filename,
            image_dir=image_full_dir,
            tokenizer=self.tokenizer,
            split="validation",
            max_items=self.max_items,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed,
            dataset_type=self.dataset_type,
        )
        num_workers = 4 if self.num_workers is None else self.num_workers
        dataloader_kwargs = {
            "dataset": val_dataset,
            "batch_size": self.batch_size,
            "shuffle": False,
            "collate_fn": custom_collate_fn,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["prefetch_factor"] = 2

        return DataLoader(**dataloader_kwargs)

    def test_dataloader(self):
        """Create test dataloader."""

        if not self.data_dir or not self.csv_filename or not self.image_dir_name:
            print(
                "Test data parameters (data_dir, csv_filename, image_dir_name) not fully specified. Skipping test."
            )
            return None

        image_full_dir = os.path.join(self.data_dir, self.image_dir_name)

        test_dataset = SemCLIPDataset(
            data_dir=self.data_dir,
            csv_filename=self.csv_filename,
            image_dir=image_full_dir,
            tokenizer=self.tokenizer,
            split="test",
            max_items=self.max_items,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed,
            dataset_type=self.dataset_type,
        )

        num_workers = 4 if self.num_workers is None else self.num_workers
        dataloader_kwargs = {
            "dataset": test_dataset,
            "batch_size": self.batch_size,
            "shuffle": False,
            "collate_fn": custom_collate_fn,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True
            dataloader_kwargs["prefetch_factor"] = 2

        return DataLoader(**dataloader_kwargs)
