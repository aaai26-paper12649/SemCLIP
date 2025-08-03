from typing import Callable, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from dataset import custom_collate_fn


def compute_topk_retrieval(image_embeddings, text_embeddings, device, batch_size=128):
    """
    Compute top-k retrieval metrics in a memory-efficient way by processing in batches.

    Args:
        image_embeddings: Normalized image embeddings (N, D)
        text_embeddings: Normalized text embeddings (N, D)
        device: Device to perform computation on
        batch_size: Size of batches to process at once to avoid OOM errors

    Returns:
        Dictionary with top-k accuracies for k in [1, 3, 5]
    """
    num_texts = text_embeddings.shape[0]
    num_images = image_embeddings.shape[0]

    if num_texts != num_images:
        raise ValueError(
            f"Number of texts ({num_texts}) must match number of images ({num_images})."
        )

    target_indices = torch.arange(num_texts, device=device)
    correct_counts = {k: 0 for k in [1, 3, 5]}

    for i in range(0, num_texts, batch_size):
        end_idx = min(i + batch_size, num_texts)
        batch_text_emb = text_embeddings[i:end_idx]

        similarity = torch.matmul(batch_text_emb, image_embeddings.T)

        for k in [1, 3, 5]:
            if k > num_images:
                correct_counts[k] = correct_counts.get(1, 0)
                continue

            _, top_k_indices = similarity.topk(k, dim=1)
            batch_target_indices = target_indices[i:end_idx].unsqueeze(1)
            correct = torch.any(top_k_indices == batch_target_indices, dim=1)
            correct_counts[k] += correct.sum().item()

    accuracies = {k: count / num_texts for k, count in correct_counts.items()}
    return accuracies


def compute_negation_retrieval(
    image_embeddings, original_text_embeddings, negation_text_embeddings, batch_size=128
):
    """
    Evaluate whether original caption embeddings align better with images than negation captions.
    Processes in batches to avoid OOM errors.

    Args:
        image_embeddings: Normalized image embeddings (N, D)
        original_text_embeddings: Normalized original caption embeddings (N, D)
        negation_text_embeddings: Normalized negation caption embeddings (N, D)
        batch_size: Size of batches to process at once to avoid OOM errors

    Returns:
        Accuracy as a float between 0 and 1
    """
    num_samples = image_embeddings.shape[0]
    correct_count = 0

    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch_img_emb = image_embeddings[i:end_idx]
        batch_orig_emb = original_text_embeddings[i:end_idx]
        batch_anti_emb = negation_text_embeddings[i:end_idx]

        sims_orig = (batch_img_emb * batch_orig_emb).sum(dim=1)
        sims_anti = (batch_img_emb * batch_anti_emb).sum(dim=1)

        correct = sims_orig > sims_anti
        correct_count += correct.sum().item()

    accuracy = correct_count / num_samples
    return accuracy


def get_embeddings(
    model, dataset, tokenizer, device="cuda", caption_type: str = "original"
):
    """Get normalized image and text embeddings for the entire dataset."""
    model.eval()
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn
    )
    image_embeddings, text_embeddings = [], []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            match caption_type:
                case "original":
                    captions = tokenizer(batch["caption"]).to(device)
                case "paraphrasing":
                    captions = tokenizer(batch["paraphrasing_caption"]).to(device)
                case "negation":
                    captions = tokenizer(batch["negated_caption"]).to(device)
                case _:
                    raise ValueError(f"Unknown caption_type: {caption_type}")
            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(captions)
            image_embeddings.append(img_emb)
            text_embeddings.append(txt_emb)
    return torch.cat(image_embeddings), torch.cat(text_embeddings)


def evaluate_caption_to_image_retrieval(
    model_name, model, dataset, tokenizer, caption_type="original"
):
    """Evaluate the model's caption-to-image retrieval performance."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        image_features, caption_features = get_embeddings(
            model, dataset, tokenizer, device, caption_type=caption_type
        )
        image_features = F.normalize(image_features, dim=-1)
        caption_features = F.normalize(caption_features, dim=-1)
        accuracies = compute_topk_retrieval(image_features, caption_features, device)
        for k, accuracy in accuracies.items():
            print(
                f"{model_name} - {caption_type} caption Top-{k} Retrieval Accuracy: {accuracy:.4f}"
            )
    return accuracies


def evaluate_original_vs_negation_retrieval(model_name, model, dataset, tokenizer):
    """Evaluate whether original caption embeddings align better with images than negation captions."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        image_embeddings, original_caption_embeddings = get_embeddings(
            model, dataset, tokenizer, device, caption_type="original"
        )
        _, negated_caption_embeddings = get_embeddings(
            model, dataset, tokenizer, device, caption_type="negation"
        )
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        original_caption_embeddings = F.normalize(original_caption_embeddings, dim=-1)
        negated_caption_embeddings = F.normalize(negated_caption_embeddings, dim=-1)
        accuracy = compute_negation_retrieval(
            image_embeddings, original_caption_embeddings, negated_caption_embeddings
        )

        print(f"{model_name} - original vs negation retrieval accuracy: {accuracy:.4f}")
    return accuracy


def evaluate_zeroshot(
    model,
    preprocess: Callable,
    tokenizer: Callable,
    dataset_name: str,
    dataset_class: Callable,
    device: str = "cuda",
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 4,
) -> Dict[str, float]:
    """
    Evaluate open_clip model on a classification dataset using zero-shot classification with original and negated prompts.

    Args:
        model: open_clip model (must have encode_image and encode_text methods)
        preprocess: Image preprocessing function (torchvision transform)
        tokenizer: Text tokenizer from open_clip.get_tokenizer()
        dataset_name: Name of the dataset to evaluate on. Supported: 'cifar10', 'cifar100', 'flowers102', 'food101', 'oxford_iiit_pet'
        device: Device to run evaluation on ('cuda' or 'cpu')
        data_dir: Directory to store/load datasets
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading

    Returns:
        dict: Dictionary with keys:
            - original_accuracy: Accuracy using original prompts (%)
            - negated_accuracy: Accuracy using negated prompts (%)
            - delta: Difference between original and negated accuracy (%)
    """
    DEFAULT_PROMPT_TEMPLATE = "this is a photo of a {}"
    DEFAULT_NEGATED_PROMPT_TEMPLATE = "this is not a photo of a {}"

    dataset = dataset_class(root=data_dir, transform=preprocess, download=True)
    classes = dataset.classes

    original_prompts = [DEFAULT_PROMPT_TEMPLATE.format(cls) for cls in classes]
    negated_prompts = [DEFAULT_NEGATED_PROMPT_TEMPLATE.format(cls) for cls in classes]

    with torch.no_grad():
        original_tokens = tokenizer(original_prompts).to(device)
        negated_tokens = tokenizer(negated_prompts).to(device)

        original_text_features = model.encode_text(original_tokens)
        negated_text_features = model.encode_text(negated_tokens)

        original_text_features = F.normalize(original_text_features, dim=-1)
        negated_text_features = F.normalize(negated_text_features, dim=-1)

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )

    model.eval()
    correct_original, correct_negated, total = 0, 0, 0

    with torch.no_grad():
        original_features = model.encode_text(original_tokens)
        original_features = F.normalize(original_features, dim=-1)

        negated_features = model.encode_text(negated_tokens)
        negated_features = F.normalize(negated_features, dim=-1)

        for images, labels in tqdm(
            test_loader, desc=f"Evaluating on {dataset_name.upper()}"
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            original_logits = (image_features @ original_text_features.t()).softmax(
                dim=-1
            )
            negated_logits = (image_features @ negated_text_features.t()).softmax(
                dim=-1
            )

            original_preds = original_logits.argmax(dim=-1)
            negated_preds = negated_logits.argmax(dim=-1)

            correct_original += (original_preds == labels).sum().item()
            correct_negated += (negated_preds == labels).sum().item()
            total += labels.size(0)

    original_accuracy = 100.0 * correct_original / total
    negated_accuracy = 100.0 * correct_negated / total
    delta = original_accuracy - negated_accuracy

    print(f"{dataset_name.upper()} Zero-shot Results:")
    print(f"  Original prompt accuracy: {original_accuracy:.2f}%")
    print(f"  Negated prompt accuracy: {negated_accuracy:.2f}%")
    print(f"  Delta (Δ): {delta:.2f}%")

    return {
        "original_accuracy": original_accuracy,
        "negated_accuracy": negated_accuracy,
        "delta": delta,
    }


def evaluate_zeroshot_on_all_datasets(
    model,
    preprocess: Callable,
    tokenizer: Callable,
    device: str = "cuda",
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 4,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate open_clip model on all supported datasets using zero-shot classification.

    Args:
        model: open_clip model (must have encode_image and encode_text methods)
        preprocess: Image preprocessing function (torchvision transform)
        tokenizer: Text tokenizer from open_clip.get_tokenizer()
        device: Device to run evaluation on ('cuda' or 'cpu')
        data_dir: Base directory to store/load datasets
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading

    Returns:
        dict: Dictionary mapping dataset names to their evaluation metrics
    """
    dataset_configs = [
        ("cifar10", datasets.CIFAR10),
        ("cifar100", datasets.CIFAR100),
        ("flowers102", datasets.Flowers102),
        ("food101", datasets.Food101),
        ("oxford_iiit_pet", datasets.OxfordIIITPet),
    ]

    results, successful_evaluations, total_original_acc, total_negated_acc = {}, [], 0.0, 0.0

    for dataset_name, dataset_class in dataset_configs:
        print(f"Evaluating on {dataset_name}...")
        try:
            metrics = evaluate_zeroshot(
                model=model,
                preprocess=preprocess,
                dataset_name=dataset_name,
                dataset_class=dataset_class,
                tokenizer=tokenizer,
                device=device,
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            results[dataset_name] = metrics
            successful_evaluations.append(dataset_name)

            if "original_accuracy" in metrics and "negated_accuracy" in metrics:
                total_original_acc += metrics["original_accuracy"]
                total_negated_acc += metrics["negated_accuracy"]

            print(f"\n{dataset_name} results:")
            for metric_name, value in metrics.items():
                unit = "%" if metric_name != "delta" else ""
                print(f"  {metric_name.replace('_', ' ').title()}: {value:.2f}{unit}")

        except Exception as e:
            print(f"Error evaluating {dataset_name}: {str(e)}")
            results[dataset_name] = {"error": str(e)}

        if (num_successful := len(successful_evaluations)) > 0:
            avg_original = total_original_acc / num_successful
            avg_negated = total_negated_acc / num_successful
            avg_delta = avg_original - avg_negated

            print(f"Average Original Accuracy: {avg_original:.2f}%")
            print(f"Average Negated Accuracy: {avg_negated:.2f}%")
            print(f"Average Delta: {avg_delta:+.2f}%")

    return results
