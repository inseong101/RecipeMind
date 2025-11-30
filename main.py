import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from visualize import (
    plot_attention_heatmap,
    plot_length_heatmap,
    plot_pmi_heatmap,
    plot_tsne_embeddings,
    set_korean_font,
)

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
    length_df = pd.DataFrame(length_rows)
    herb_df = pd.DataFrame(herb_rows)
    length_df.to_csv(os.path.join("results", "tables", "length_accuracy.csv"), index=False)
    herb_df.to_csv(os.path.join("results", "tables", "herb_accuracy.csv"), index=False)
    return length_df, herb_df


def compute_pmi(preprocessed: Dict) -> np.ndarray:
    co_matrix = preprocessed["co_matrix"]
    herb2id = preprocessed["herb2id"]
    herb_counts = preprocessed["herb_counts"]
    freq = np.zeros(len(herb2id), dtype=np.float64)
    for herb, idx in herb2id.items():
        freq[idx] = herb_counts.get(herb, 0)

    freq_sum = freq.sum() + 1e-8
    pair_sum = co_matrix.sum() / 2 + 1e-8
    pmi = np.zeros_like(co_matrix, dtype=np.float64)
    for i in range(len(herb2id)):
        for j in range(len(herb2id)):
            if i == j:
                continue
            p_i = freq[i] / freq_sum
            p_j = freq[j] / freq_sum
            p_ij = co_matrix[i, j] / pair_sum if pair_sum > 0 else 0.0
            if p_ij > 0 and p_i > 0 and p_j > 0:
                pmi[i, j] = np.log(p_ij / (p_i * p_j))
    return pmi


def save_summary(summary: Dict):
    json.dump(summary, open(RESULT_SUMMARY, "w"), ensure_ascii=False, indent=2)


def load_checkpoint(model: HerbMindModel):
    if os.path.exists(CHECKPOINT_PATH):
        state = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state)
        return True
    return False


def collect_attention_example(model: HerbMindModel, preprocessed: Dict, device: torch.device):
    example = preprocessed["prescriptions"][0]
    herb2id = preprocessed["herb2id"]
    id2herb = preprocessed["id2herb"]
    herb_ids = [herb2id[h] for h in example["herbs"] if h in herb2id]
    if len(herb_ids) < 2:
        return None

    context = herb_ids[:-1][:6]
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    mask = torch.tensor([[1] * len(context)], dtype=torch.float, device=device)
    logits, attn_maps = model(input_ids, mask, return_attn=True)
    probs = torch.softmax(logits, dim=-1)[0]
    top_candidates = torch.topk(probs, k=min(8, probs.shape[0]))
    cross_attn = attn_maps[-1][0].detach().cpu().numpy()
    rows = [id2herb[idx.item()] for idx in top_candidates.indices]
    cols = [id2herb[i] for i in context]
    attn_sub = cross_attn[top_candidates.indices.cpu().numpy()][:, : len(context)]
    return attn_sub, rows, cols


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
    set_korean_font()

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

    pmi_matrix = compute_pmi(preprocessed)
    plot_length_heatmap(length_df, os.path.join("results", "figures", "figure_length_heatmap.png"))
    plot_tsne_embeddings(model, preprocessed["id2herb"], os.path.join("results", "figures", "figure_tsne_embeddings.png"))
    plot_pmi_heatmap(pmi_matrix, preprocessed["id2herb"], os.path.join("results", "figures", "figure_pmi_heatmap.png"))

    attention_payload = collect_attention_example(model, preprocessed, device)
    if attention_payload:
        attn, rows, cols = attention_payload
        plot_attention_heatmap(attn, rows, cols, os.path.join("results", "figures", "figure_attention_heatmap.png"))

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
        "pmi_max": float(np.max(pmi_matrix)) if pmi_matrix.size else 0.0,
    }
    save_summary(summary)
    print("Pipeline completed. Results saved to results/ directory.")


if __name__ == "__main__":
    main()
