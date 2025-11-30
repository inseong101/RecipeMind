import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment fallback
    plt = None
    sns = None
    PLOTTING_AVAILABLE = False
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from tqdm import tqdm

from herbmind_data import (
    PREPROCESSED_PATH,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
    ensure_dirs,
    generate_splits,
    load_and_preprocess,
)
from herbmind_dataset import build_dataloaders
from model import HerbMindModel


RESULT_SUMMARY = os.path.join("results", "summary.json")
CHECKPOINT_PATH = os.path.join("checkpoints", "herbmind.pt")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for inputs, masks, targets, _ in tqdm(loader, desc="Train"):
        inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs, masks)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
    return total_loss / total, correct / total


def evaluate(model, loader, device, presc_lengths: Dict[int, int]):
    model.eval()
    total = 0
    correct1 = 0
    correct3 = 0
    herb_stats: Dict[int, Dict[str, float]] = {}
    length_stats: Dict[int, Dict[str, float]] = {}

    with torch.no_grad():
        for inputs, masks, targets, presc_ids in tqdm(loader, desc="Eval"):
            inputs, masks, targets, presc_ids = (
                inputs.to(device),
                masks.to(device),
                targets.to(device),
                presc_ids.numpy(),
            )
            logits = model(inputs, masks)
            top3 = torch.topk(logits, k=3, dim=-1).indices.cpu().numpy()
            preds1 = top3[:, 0]
            targets_np = targets.cpu().numpy()
            total += targets_np.shape[0]
            correct1 += (preds1 == targets_np).sum()
            correct3 += sum(target in row for target, row in zip(targets_np, top3))

            for target in targets_np:
                if target not in herb_stats:
                    herb_stats[target] = {"count": 0, "top1": 0, "top3": 0}
                herb_stats[target]["count"] += 1
            for target, row in zip(targets_np, top3):
                herb_stats[target]["top1"] += int(target == row[0])
                herb_stats[target]["top3"] += int(target in row)

            for pid, row in zip(presc_ids, top3):
                length = presc_lengths.get(pid, inputs.shape[1])
                if length not in length_stats:
                    length_stats[length] = {"count": 0, "top1": 0, "top3": 0}
                length_stats[length]["count"] += 1
            for pid, target, row in zip(presc_ids, targets_np, top3):
                length = presc_lengths.get(pid, inputs.shape[1])
                length_stats[length]["top1"] += int(target == row[0])
                length_stats[length]["top3"] += int(target in row)

    metrics = {
        "top1": correct1 / total if total else 0.0,
        "top3": correct3 / total if total else 0.0,
        "total": total,
    }
    return metrics, herb_stats, length_stats


def length_table(length_stats: Dict[int, Dict[str, float]]):
    rows = []
    for length, stats in sorted(length_stats.items()):
        count = stats["count"]
        rows.append(
            {
                "length": length,
                "count": count,
                "top1": stats["top1"] / count if count else 0.0,
                "top3": stats["top3"] / count if count else 0.0,
            }
        )
    return rows


def herb_table(herb_stats: Dict[int, Dict[str, float]], id2herb: Dict[int, str], herb_counts: Dict[str, int]):
    rows = []
    for herb_id, stats in herb_stats.items():
        count = stats["count"]
        name = id2herb.get(herb_id, str(herb_id))
        rows.append(
            {
                "herb_id": herb_id,
                "herb_name": name,
                "dataset_frequency": herb_counts.get(name, 0),
                "eval_count": count,
                "top1": stats["top1"] / count if count else 0.0,
                "top3": stats["top3"] / count if count else 0.0,
            }
        )
    rows.sort(key=lambda x: x["dataset_frequency"], reverse=True)
    return rows


def save_tables(length_rows: List[Dict], herb_rows: List[Dict]):
    import pandas as pd

    length_df = pd.DataFrame(length_rows)
    herb_df = pd.DataFrame(herb_rows)
    length_df.to_csv(os.path.join("results", "tables", "length_accuracy.csv"), index=False)
    herb_df.to_csv(os.path.join("results", "tables", "herb_accuracy.csv"), index=False)
    return length_df, herb_df


def plot_length_heatmap(length_df):
    if length_df.empty or not PLOTTING_AVAILABLE:
        return
    bins = [2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf]
    labels = ["2", "3", "4", "5", "6", "7", "8", "9", "10+"]
    length_df["bin"] = pd.cut(length_df["length"], bins=bins, labels=labels, right=False)
    pivot = length_df.groupby("bin")["top1"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    sns.heatmap(pivot.set_index("bin"), annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Figure A: Accuracy by Prescription Length")
    plt.ylabel("Length Bin")
    plt.xlabel("Top-1 Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join("results", "figures", "figure_length_heatmap.png"))
    plt.close()


def plot_frequency_scatter(herb_df):
    if herb_df.empty or not PLOTTING_AVAILABLE:
        return
    plt.figure(figsize=(8, 5))
    plt.scatter(herb_df["dataset_frequency"], herb_df["top1"], alpha=0.6)
    plt.xlabel("Herb Frequency")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Figure B: Herb Frequency vs Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join("results", "figures", "figure_frequency_scatter.png"))
    plt.close()


def plot_tsne_embeddings(model: HerbMindModel, id2herb: Dict[int, str]):
    if not PLOTTING_AVAILABLE:
        return
    embeddings = model.embed.weight.detach().cpu().numpy()[:-1]  # exclude padding
    if embeddings.shape[0] < 2:
        return
    perplexity = min(30, embeddings.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42)
    coords = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.7)
    for idx, (x, y) in enumerate(coords):
        if idx % max(1, embeddings.shape[0] // 50) == 0:
            plt.text(x, y, id2herb.get(idx, str(idx)), fontsize=6)
    plt.title("Figure C: t-SNE of Herb Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join("results", "figures", "figure_tsne_embeddings.png"))
    plt.close()


def plot_attention_example(model: HerbMindModel, preprocessed: Dict):
    if not preprocessed["prescriptions"] or not PLOTTING_AVAILABLE:
        return
    device = next(model.parameters()).device
    example = preprocessed["prescriptions"][0]
    herb2id = preprocessed["herb2id"]
    id2herb = preprocessed["id2herb"]
    herb_ids = [herb2id[h] for h in example["herbs"] if h in herb2id]
    if len(herb_ids) < 2:
        return

    sequence = [herb_ids[0]]
    recommendations = []
    for _ in range(min(3, len(herb_ids) - 1)):
        mask = torch.tensor([[1] * len(sequence)], dtype=torch.float, device=device)
        input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
        logits, attn_maps = model(input_ids, mask, return_attn=True)
        probs = torch.softmax(logits, dim=-1)
        top3 = torch.topk(probs, k=3, dim=-1).indices[0].cpu().numpy()
        recommendations.append([id2herb.get(i, str(i)) for i in top3])
        sequence.append(herb_ids[len(sequence)])

    if attn_maps:
        attn = attn_maps[-1][0].detach().cpu().numpy()
        plt.figure(figsize=(6, 4))
        sns.heatmap(attn, cmap="OrRd")
        plt.xlabel("Key/Value Herbs")
        plt.ylabel("Query Herbs")
        plt.title("Figure D: Attention Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join("results", "figures", "figure_attention_heatmap.png"))
        plt.close()

    table_path = os.path.join("results", "tables", "example_recommendations.csv")
    import pandas as pd

    pd.DataFrame({"step": list(range(1, len(recommendations) + 1)), "top3": recommendations}).to_csv(
        table_path, index=False
    )


def save_summary(summary: Dict):
    json.dump(summary, open(RESULT_SUMMARY, "w"), ensure_ascii=False, indent=2)


def load_checkpoint(model: HerbMindModel):
    if os.path.exists(CHECKPOINT_PATH):
        state = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="HerbMind turnkey pipeline")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_train", action="store_true", help="retrain even if checkpoint exists")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)

    preprocessed = load_and_preprocess()
    generate_splits(preprocessed, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HerbMindModel(
        num_herbs=len(preprocessed["herb2id"]),
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        num_blocks=args.blocks,
    ).to(device)

    checkpoint_loaded = False
    if not args.force_train:
        checkpoint_loaded = load_checkpoint(model)

    loaders = build_dataloaders(TRAIN_PATH, VAL_PATH, TEST_PATH, batch_size=args.batch_size)
    presc_lengths = {p["id"]: len(p["herbs"]) for p in preprocessed["prescriptions"]}

    history = []
    if not checkpoint_loaded:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        best_val = 0.0
        for epoch in range(1, args.epochs + 1):
            loss, acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
            val_metrics, _, _ = evaluate(model, loaders["val"], device, presc_lengths)
            history.append({"epoch": epoch, "loss": loss, "train_acc": acc, "val_top1": val_metrics["top1"]})
            print(f"Epoch {epoch}: loss={loss:.4f}, train_acc={acc:.4f}, val_top1={val_metrics['top1']:.4f}")
            if val_metrics["top1"] > best_val:
                best_val = val_metrics["top1"]
                torch.save(model.state_dict(), CHECKPOINT_PATH)
        load_checkpoint(model)
    else:
        print("Checkpoint loaded, skipping training. Use --force_train to retrain.")

    val_metrics, val_herb_stats, val_length_stats = evaluate(model, loaders["val"], device, presc_lengths)
    test_metrics, herb_stats, length_stats = evaluate(model, loaders["test"], device, presc_lengths)

    length_rows = length_table(length_stats)
    herb_rows = herb_table(herb_stats, preprocessed["id2herb"], preprocessed["herb_counts"])
    length_df, herb_df = save_tables(length_rows, herb_rows)

    plot_length_heatmap(length_df)
    plot_frequency_scatter(herb_df)
    plot_tsne_embeddings(model, preprocessed["id2herb"])
    plot_attention_example(model, preprocessed)

    summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "params": vars(args),
        "num_prescriptions": len(preprocessed["prescriptions"]),
        "num_herbs": len(preprocessed["herb2id"]),
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "history": history,
    }
    save_summary(summary)
    print("Pipeline completed. Results saved to results/ directory.")


if __name__ == "__main__":
    main()
