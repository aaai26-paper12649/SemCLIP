import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch):
    """
    Collate function for batches with images and three caption types.
    Returns a dict with stacked images and padded captions.
    """
    images = torch.stack([item["image"] for item in batch])
    caption_keys = ["caption", "negated_caption", "paraphrased_caption"]
    captions = {}
    for key in caption_keys:
        seqs = [item[key] for item in batch]
        captions[key] = pad_sequence(seqs, batch_first=True, padding_value=0)
    return {"image": images, **captions}


class SemCLIPDataset(Dataset):
    """Dataset for generic images with original, paraphrasing, and negation captions, loaded from a CSV file."""

    def __init__(
        self,
        data_dir: str,
        csv_filename: str,
        image_dir: str,
        image_extension: str = ".jpg",
        use_imagenet_transforms: bool = True,
        image_size: tuple = (224, 224),
        tokenizer: callable = None,
        split: str = "train",
        max_items: int = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        dataset_type: str = "ccneg",
    ):
        """Initialize the dataset.

        Args:
            data_dir: Path to the directory containing the CSV file.
            csv_filename: Name of the CSV file (e.g., 'ccneg_paraphrased.csv').
            image_dir: Path to the directory containing the images.
            image_extension: File extension for images (e.g., '.jpg', '.png').
            use_imagenet_transforms: Whether to use ImageNet normalization.
            image_size: Size to resize images to.
            tokenizer: Callable tokenizer for captions.
            split: Dataset split to use ('train', 'validation', or 'test').
            max_items: Optional maximum number of items to load from the dataset.
            val_ratio: Ratio of data to use for validation (default: 0.1).
            test_ratio: Ratio of data to use for test (default: 0.1).
            random_seed: Random seed for reproducible data splits.
            dataset_type: Type of dataset ('ccneg' or 'sugarcrepe'). If None, will attempt to auto-detect.
        """
        self.data_dir = Path(data_dir)
        self.csv_filename = csv_filename
        self.image_dir = Path(image_dir)
        self.image_extension = image_extension
        self.tokenizer = tokenizer
        self.split = split
        self.max_items = max_items
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.dataset_type = dataset_type

        self.dataset = self.load_dataset_from_csv()

        if use_imagenet_transforms:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )

    def load_dataset_from_csv(self):
        """Load the dataset from a CSV file and split into train/validation/test sets."""
        csv_path = self.data_dir / self.csv_filename
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file {csv_path}: {e}")

        if self.max_items is not None and self.max_items > 0:
            df = df.head(self.max_items)

        np.random.seed(self.random_seed)

        df = df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)

        total_size = len(df)
        test_size = int(total_size * self.test_ratio)
        val_size = int(total_size * self.val_ratio)
        train_size = total_size - test_size - val_size

        match self.split:
            case "train":
                split_df = df.iloc[:train_size]
            case "validation":
                split_df = df.iloc[train_size : train_size + val_size]
            case "test":
                split_df = df.iloc[train_size + val_size :]
            case _:
                raise ValueError(
                    f"Invalid split: {self.split}. Must be 'train', 'validation', or 'test'."
                )

        return split_df.to_dict("records")

    def __getitem__(self, idx):
        """Get a single item from the dataset.

        Returns a dictionary with:
            - image: Transformed image tensor
            - caption: Original caption
            - paraphrased_caption: Paraphrased caption
            - negated_caption: Negated caption
        """
        item = self.dataset[idx]

        match self.dataset_type:
            case "ccneg":
                image_id_str = str(item["image_number"]).zfill(9)
                image_filename = f"{image_id_str}{self.image_extension}"
            case "sugarcrepe":
                image_filename = str(item["filename"])
            case _:
                raise ValueError(
                    f"Unable to determine dataset format. Please specify dataset_type='ccneg' or 'sugarcrepe'. "
                )

        image_path = self.image_dir / image_filename

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path} for item index {idx}")
            raise FileNotFoundError(f"Image file not found at {image_path}")

        caption_text = item["caption"]
        negated_caption_text = item["negation"]
        paraphrased_caption_text = item["paraphrased"]

        caption_tokens = self.tokenizer(caption_text)[0]
        paraphrase_tokens = self.tokenizer(paraphrased_caption_text)[0]
        negation_tokens = self.tokenizer(negated_caption_text)[0]

        return {
            "image": image,
            "caption": caption_tokens,
            "paraphrased_caption": paraphrase_tokens,
            "negated_caption": negation_tokens,
        }

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.dataset)
