import os
from itertools import chain
from typing import Dict, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, DownloadMode, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(name: str = "gpt2") -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(1e6)
    return tokenizer


def _tokenize_function(examples: Dict[str, list], tokenizer: PreTrainedTokenizerBase) -> Dict[str, list]:
    return tokenizer(examples["text"])


def _group_texts(examples: Dict[str, list], seq_len: int) -> Dict[str, list]:
    # Concatenate within a batch then split into blocks
    concatenated = list(chain.from_iterable(examples["input_ids"]))
    total_length = (len(concatenated) // seq_len) * seq_len
    concatenated = concatenated[:total_length]
    result = {"input_ids": [concatenated[i : i + seq_len] for i in range(0, total_length, seq_len)]}
    return result


def prepare_datasets(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    seed: int,
    val_split: str = "validation",
    val_fraction: float = 0.01,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    num_proc: int = 1,
    cache_dir: str = "data/cache",
    dataset_variant: str = "full",
    small_train_samples: int = 50_000,
    small_val_samples: int = 5_000,
) -> Tuple[Dataset, Dataset]:
    dataset_variant = dataset_variant.lower()
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    raw: DatasetDict = load_dataset(
        dataset_name,
        cache_dir=cache_dir,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )

    if val_split in raw:
        train_raw = raw["train"]
        val_raw = raw[val_split]
    else:
        split = raw["train"].train_test_split(test_size=val_fraction, seed=seed)
        train_raw, val_raw = split["train"], split["test"]

    if dataset_variant == "small":
        train_raw = train_raw.select(range(min(len(train_raw), small_train_samples)))
        val_raw = val_raw.select(range(min(len(val_raw), small_val_samples)))
    elif dataset_variant != "full":
        raise ValueError(f"dataset_variant must be 'full' or 'small', got '{dataset_variant}'")

    if max_train_samples:
        train_raw = train_raw.select(range(min(len(train_raw), max_train_samples)))
    if max_val_samples:
        val_raw = val_raw.select(range(min(len(val_raw), max_val_samples)))

    tokenized_train = train_raw.map(
        lambda batch: _tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=train_raw.column_names,
        num_proc=num_proc,
    )
    tokenized_val = val_raw.map(
        lambda batch: _tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=val_raw.column_names,
        num_proc=num_proc,
    )

    grouped_train = tokenized_train.map(
        lambda batch: _group_texts(batch, seq_len),
        batched=True,
        num_proc=num_proc,
    )
    grouped_val = tokenized_val.map(
        lambda batch: _group_texts(batch, seq_len),
        batched=True,
        num_proc=num_proc,
    )

    def set_format(ds: Dataset) -> Dataset:
        return ds.with_format(type="torch", columns=["input_ids"])

    return set_format(grouped_train), set_format(grouped_val)


def collate_batch(batch):
    # batch is list of dicts with 'input_ids'
    input_ids = torch.stack([item["input_ids"] for item in batch])
    return {"input_ids": input_ids}


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_batch,
    )
