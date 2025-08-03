import os

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from custom_loss import NegationParaphraseProjectionLoss
from dataset import SemCLIPDataset, custom_collate_fn
from evaluation import (
    compute_negation_retrieval,
    compute_topk_retrieval,
    evaluate_zeroshot_on_all_datasets,
)

class SemCLIP(L.LightningModule):
    def __init__(self, training_params):
        super().__init__()
        self.save_hyperparameters(ignore=["training_params"])
        self.training_params = training_params
        self.model_name = training_params.model_name
        self.pretrained = training_params.pretrained
        self.image_dir = os.path.join(
            self.training_params.data_dir, self.training_params.image_dir_name
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / training_params.temperature))
        )

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

    @property
    def learning_rate(self):
        return self.training_params.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.training_params.learning_rate = value

    def training_step(self, batch, batch_idx):
        """Training step."""
        images = batch["image"]

        cap_tokens = batch["caption"].to(self.training_params.device)
        paraphrasing_tokens = batch["paraphrased_caption"].to(
            self.training_params.device
        )
        negation_tokens = batch["negated_caption"].to(self.training_params.device)

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

        for loss_component, value in loss.items():
            self.log(
                f"train_{loss_component}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=loss_component == "loss",
            )

        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch["image"]

        cap_tokens = batch["caption"].to(self.training_params.device)
        paraphrasing_tokens = batch["paraphrased_caption"].to(
            self.training_params.device
        )
        negation_tokens = batch["negated_caption"].to(self.training_params.device)

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

        for loss_component in loss.keys():
            result[loss_component] = loss[loss_component]
            self.log(
                f"val_{loss_component}",
                loss[loss_component],
                on_step=False,
                on_epoch=True,
                prog_bar=loss_component == "loss",
            )

        self.validation_step_outputs.append(result)

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
                    device=self.training_params.device,
                    data_dir=self.training_params.data_dir,
                    batch_size=self.training_params.batch_size,
                    num_workers=self.training_params.num_workers,
                )

                total_original_acc, total_negated_acc = 0.0, 0.0
                num_datasets = len(all_metrics)

                for dataset_name, metrics in all_metrics.items():
                    total_original_acc += metrics["original_accuracy"]
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

                avg_original = total_original_acc / num_datasets
                avg_negated = total_negated_acc / num_datasets
                avg_delta = avg_original - avg_negated

                for metric_name, value in [
                    ("original_accuracy", avg_original),
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

        all_image_emb = F.normalize(
            torch.cat([x["image_emb"] for x in outputs]), dim=-1
        )
        all_original_emb = F.normalize(
            torch.cat([x["text_emb"] for x in outputs]), dim=-1
        )
        all_paraphrasing_emb = F.normalize(
            torch.cat([x["paraphrasing_emb"] for x in outputs]), dim=-1
        )
        all_negation_emb = F.normalize(
            torch.cat([x["negation_emb"] for x in outputs]), dim=-1
        )

        topk_original = compute_topk_retrieval(
            all_image_emb,
            all_original_emb,
            self.training_params.device,
            batch_size=self.training_params.batch_size,
        )

        topk_paraphrasing = compute_topk_retrieval(
            all_image_emb,
            all_paraphrasing_emb,
            self.training_params.device,
            batch_size=self.training_params.batch_size,
        )

        accuracy = compute_negation_retrieval(
            all_image_emb,
            all_original_emb,
            all_negation_emb,
            batch_size=self.training_params.batch_size,
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
        self.log(
            f"{prefix}_combined_score",
            (topk_original[1] + topk_paraphrasing[1] + scaled_accuracy) / 3.0,
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

    def get_dataloader(self, split):
        """Create a DataLoader for the specified data split (train, validation, or test)"""
        dataset = SemCLIPDataset(
            data_dir=self.training_params.data_dir,
            csv_filename=self.training_params.csv_filename,
            image_dir=self.image_dir,
            tokenizer=self.tokenizer,
            split=split,
            max_items=self.training_params.max_items,
            random_seed=self.training_params.random_seed,
            dataset_type=self.training_params.dataset_type,
        )
        num_workers = self.training_params.num_workers
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": self.training_params.batch_size,
            "shuffle": split == "train",
            "collate_fn": custom_collate_fn,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 2
        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self):
        return self.get_dataloader(split="train")

    def val_dataloader(self):
        return self.get_dataloader(split="validation")

    def test_dataloader(self):
        return self.get_dataloader(split="test")