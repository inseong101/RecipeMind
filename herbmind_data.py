import itertools
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PREPROCESSED_PATH = os.path.join("data", "preprocessed_prescriptions.pkl")
TRAIN_PATH = os.path.join("data", "train.pkl")
VAL_PATH = os.path.join("data", "val.pkl")
TEST_PATH = os.path.join("data", "test.pkl")


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    os.makedirs(os.path.join("results", "tables"), exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)


def read_kiom_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def load_and_preprocess(data_dir: str = "KIOM") -> Dict:
    ensure_dirs()
    if os.path.exists(PREPROCESSED_PATH):
        return pickle.load(open(PREPROCESSED_PATH, "rb"))

    all_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    ]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for path in all_files:
        df = read_kiom_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        if "처방명" not in df.columns or "약재명" not in df.columns:
            raise ValueError(f"Required columns missing in {path}. Columns: {df.columns}")
        frames.append(df[["처방명", "약재명"]].copy())
    merged = pd.concat(frames, ignore_index=True)
    merged["처방명"] = merged["처방명"].astype(str).str.strip()
    merged["약재명"] = merged["약재명"].astype(str).str.strip()

    grouped = merged.groupby("처방명")
    prescriptions: List[Dict] = []
    herb_counts = merged["약재명"].value_counts().to_dict()
    herb2id: Dict[str, int] = {}

    for idx, (name, group) in enumerate(grouped):
        herbs = list(dict.fromkeys(group["약재명"].tolist()))
        if len(herbs) < 2:
            continue
        prescriptions.append({
            "id": idx,
            "name": name,
            "herbs": herbs,
        })
        for herb in herbs:
            if herb not in herb2id:
                herb2id[herb] = len(herb2id)

    if not prescriptions:
        raise ValueError("No valid prescriptions with at least two herbs were found.")

    id2herb = {v: k for k, v in herb2id.items()}
    preprocessed = {
        "prescriptions": prescriptions,
        "herb2id": herb2id,
        "id2herb": id2herb,
        "herb_counts": herb_counts,
    }
    pickle.dump(preprocessed, open(PREPROCESSED_PATH, "wb"))
    return preprocessed


def create_quiz_samples(prescription: Dict, herb2id: Dict[str, int]) -> List[Tuple[List[int], int, int]]:
    herbs = [herb2id[h] for h in prescription["herbs"] if h in herb2id]
    samples = []
    for i, target in enumerate(herbs):
        context = herbs[:i] + herbs[i + 1 :]
        if not context:
            continue
        samples.append((context, target, prescription["id"]))
    return samples


def generate_splits(preprocessed: Dict, seed: int = 42, test_ratio: float = 0.1, val_ratio: float = 0.1):
    if all(os.path.exists(p) for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]):
        return

    prescriptions = preprocessed["prescriptions"]
    train_presc, temp = train_test_split(prescriptions, test_size=test_ratio + val_ratio, random_state=seed, shuffle=True)
    relative_val = val_ratio / (test_ratio + val_ratio)
    val_presc, test_presc = train_test_split(temp, test_size=1 - relative_val, random_state=seed, shuffle=True)

    splits = {
        "train": train_presc,
        "val": val_presc,
        "test": test_presc,
    }

    for split_name, presc_list in splits.items():
        records = list(
            itertools.chain.from_iterable(
                create_quiz_samples(presc, preprocessed["herb2id"]) for presc in presc_list
            )
        )
        pickle.dump(records, open({"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}[split_name], "wb"))

