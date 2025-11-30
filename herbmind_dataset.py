import pickle
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class HerbQuizDataset(Dataset):
    def __init__(self, path: str):
        self.samples: List[Tuple[List[int], int, int]] = pickle.load(open(path, "rb"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        context, target, presc_id = self.samples[idx]
        return context, target, presc_id


def collate_batch(batch, pad_token: int = 0):
    contexts, targets, presc_ids = zip(*batch)
    max_len = max(len(ctx) for ctx in contexts)
    padded = []
    masks = []
    for ctx in contexts:
        padding = [pad_token] * (max_len - len(ctx))
        padded.append(ctx + padding)
        masks.append([1] * len(ctx) + [0] * (max_len - len(ctx)))
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(masks, dtype=torch.float),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(presc_ids, dtype=torch.long),
    )


def build_dataloaders(train_path: str, val_path: str, test_path: str, batch_size: int = 64, num_workers: int = 0):
    loaders = {}
    for split, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        dataset = HerbQuizDataset(path)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_batch,
            num_workers=num_workers,
        )
    return loaders
