import argparse
import json
import math
import os
import pickle
import random
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    cross_attn = attn_maps[-1]
    if cross_attn.dim() == 4:
        # (batch, heads, tgt_len, src_len) -> average heads, select batch 0
        cross_attn = cross_attn.mean(dim=1)[0]
    elif cross_attn.dim() == 3:
        # (batch, tgt_len, src_len) -> select batch 0
        cross_attn = cross_attn[0]
    cross_attn = cross_attn.detach().cpu().numpy()
    rows = [id2herb[idx.item()] for idx in top_candidates.indices]
    cols = [id2herb[i] for i in context]
    attn_sub = cross_attn[top_candidates.indices.cpu().numpy()][:, : len(context)]
    return attn_sub, rows, cols


def load_split_prescription_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split_name, path in [("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)]:
        if not os.path.exists(path):
            counts[split_name] = 0
            continue
        records = pickle.load(open(path, "rb"))
        ids = {rec[2] for rec in records}
        counts[split_name] = len(ids)
    return counts


def compute_dataset_stats(preprocessed: Dict) -> pd.DataFrame:
    prescriptions = preprocessed["prescriptions"]
    total_presc = len(prescriptions)
    total_herb_occurrences = sum(len(p["herbs"]) for p in prescriptions)
    unique_herbs = len(preprocessed["herb2id"])
    length_counts: Dict[int, int] = {}
    for p in prescriptions:
        length_counts[len(p["herbs"])] = length_counts.get(len(p["herbs"]), 0) + 1

    split_counts = load_split_prescription_counts()

    rows = [
        {"Statistic": "Total prescriptions", "Value": total_presc},
        {"Statistic": "Unique herbs", "Value": unique_herbs},
        {"Statistic": "Total herb occurrences", "Value": total_herb_occurrences},
    ]

    for length in range(2, 8):
        rows.append(
            {
                "Statistic": f"Prescriptions of length {length}",
                "Value": length_counts.get(length, 0),
            }
        )

    rows.extend(
        [
            {"Statistic": "Train set size", "Value": split_counts.get("train", 0)},
            {"Statistic": "Validation set size", "Value": split_counts.get("val", 0)},
            {"Statistic": "Test set size", "Value": split_counts.get("test", 0)},
        ]
    )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("results", "tables", "table_dataset_stats.csv"), index=False)
    return df


def build_herb_index(prescriptions: Sequence[Dict]) -> Dict[str, set]:
    herb_to_presc: Dict[str, set] = {}
    for presc in prescriptions:
        pid = presc["id"]
        for herb in presc["herbs"]:
            herb_to_presc.setdefault(herb, set()).add(pid)
    return herb_to_presc


def compute_spmi_values(preprocessed: Dict) -> Tuple[Dict[Tuple[Tuple[str, ...], str], Tuple[float, int]], Dict[int, List[float]]]:
    prescriptions = preprocessed["prescriptions"]
    total_presc = len(prescriptions)
    herb_to_presc = build_herb_index(prescriptions)

    @lru_cache(maxsize=None)
    def intersection_size(herb_tuple: Tuple[str, ...]) -> int:
        if not herb_tuple:
            return 0
        sets = [herb_to_presc.get(h, set()) for h in herb_tuple]
        if not all(sets):
            return 0
        inter = sets[0]
        for s in sets[1:]:
            inter = inter & s
            if not inter:
                break
        return len(inter)

    spmi_by_context_len: Dict[int, List[float]] = {i: [] for i in range(2, 8)}
    combo_scores: Dict[Tuple[Tuple[str, ...], str], Tuple[float, int]] = {}

    for presc in prescriptions:
        herbs = list(dict.fromkeys(presc["herbs"]))
        for herb in herbs:
            context = [h for h in herbs if h != herb]
            if not (2 <= len(context) <= 7):
                continue
            context_tuple = tuple(sorted(context))
            full_tuple = tuple(sorted(context + [herb]))
            freq_s = intersection_size(context_tuple)
            freq_h = len(herb_to_presc.get(herb, set()))
            freq_sh = intersection_size(full_tuple)
            if freq_s == 0 or freq_h == 0 or freq_sh == 0:
                continue
            p_s = freq_s / total_presc
            p_h = freq_h / total_presc
            p_sh = freq_sh / total_presc
            if p_s <= 0 or p_h <= 0 or p_sh <= 0:
                continue
            spmi = math.log2(p_sh / (p_s * p_h))
            spmi_by_context_len[len(context)].append(spmi)
            key = (context_tuple, herb)
            if key not in combo_scores:
                combo_scores[key] = (spmi, freq_sh)

    return combo_scores, spmi_by_context_len


def save_pmi_examples(combo_scores: Dict[Tuple[Tuple[str, ...], str], Tuple[float, int]]):
    if not combo_scores:
        return pd.DataFrame()
    sorted_combos = sorted(combo_scores.items(), key=lambda kv: kv[1][0])
    extremes = sorted_combos[:2] + sorted_combos[-2:]
    rows = []
    for (context, herb), (score, freq) in extremes:
        rows.append(
            {
                "context_herbs": list(context),
                "added_herb": herb,
                "sPMI": round(score, 2),
                "frequency": freq,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("results", "tables", "table_pmi_examples.csv"), index=False)
    return df


def plot_spmi_distribution(spmi_by_context_len: Dict[int, List[float]]):
    plt.figure(figsize=(8, 6))
    for size in range(2, 8):
        values = spmi_by_context_len.get(size, [])
        if not values:
            continue
        sns.kdeplot(values, label=f"|S|={size}", bw_adjust=1.2)
    plt.xlabel("sPMI")
    plt.ylabel("Density")
    plt.title("sPMI Score Distribution by Context Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "figures", "figure_spmi_distribution.png"))
    plt.close()


def predict_topk_names(model: HerbMindModel, device: torch.device, context_ids: List[int], id2herb: Dict[int, str], topk: int = 3):
    if not context_ids:
        input_ids = torch.tensor([[model.pad_id]], dtype=torch.long, device=device)
        mask = torch.tensor([[0.0]], dtype=torch.float, device=device)
    else:
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        mask = torch.tensor([[1.0] * len(context_ids)], dtype=torch.float, device=device)

    with torch.no_grad():
        logits = model(input_ids, mask)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    excluded = set(context_ids)
    ordered = np.argsort(probs)[::-1]
    top_indices = []
    for idx in ordered:
        if idx in excluded:
            continue
        top_indices.append(idx)
        if len(top_indices) == topk:
            break
    return [id2herb[i] for i in top_indices]


def extract_attention_matrix(attn_maps: List[torch.Tensor], context_len: int) -> np.ndarray:
    if not attn_maps:
        return np.zeros((context_len, context_len), dtype=np.float32)
    attn = attn_maps[-2] if len(attn_maps) >= 2 else attn_maps[-1]
    if attn.dim() == 4:
        # Shape: (batch, heads, tgt_len, src_len) when average_attn_weights=False
        attn_mean = attn.mean(dim=1)[0]
    elif attn.dim() == 3:
        # Shape: (batch, tgt_len, src_len) when attention is already head-averaged
        attn_mean = attn[0]
    elif attn.dim() == 2:
        # Shape: (tgt_len, src_len) if batch/head dimensions were removed upstream
        attn_mean = attn
    else:
        return np.zeros((context_len, context_len), dtype=np.float32)

    attn_np = attn_mean.detach().cpu().numpy()
    return attn_np[:context_len, :context_len]


def accumulate_attention(
    model: HerbMindModel,
    device: torch.device,
    context_ids: List[int],
    final_len: int,
) -> np.ndarray:
    if not context_ids:
        return np.zeros((final_len, final_len), dtype=np.float32)
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    mask = torch.tensor([[1.0] * len(context_ids)], dtype=torch.float, device=device)
    with torch.no_grad():
        _, attn_maps = model(input_ids, mask, return_attn=True)
    attn_matrix = extract_attention_matrix(attn_maps, len(context_ids))
    accum = np.zeros((final_len, final_len), dtype=np.float32)
    accum[: len(context_ids), : len(context_ids)] += attn_matrix
    return accum


def run_sequential_example(
    model: HerbMindModel,
    device: torch.device,
    preprocessed: Dict,
    herbs_sequence: Sequence[str],
    table_path: str,
    figure_path: str,
):
    herb2id = preprocessed["herb2id"]
    id2herb = preprocessed["id2herb"]
    context_ids: List[int] = []
    rows = []
    final_len = len(herbs_sequence)
    attention_accum = np.zeros((final_len, final_len), dtype=np.float32)

    for step, herb_name in enumerate(herbs_sequence, start=1):
        top3 = predict_topk_names(model, device, context_ids, id2herb, topk=3)
        rows.append({"step": step, "top3": str(top3)})

        herb_id = herb2id.get(herb_name)
        if herb_id is None:
            continue
        context_ids.append(herb_id)
        attention_accum += accumulate_attention(model, device, context_ids, final_len)

    pd.DataFrame(rows).to_csv(table_path, index=False)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        attention_accum,
        cmap="OrRd",
        xticklabels=herbs_sequence,
        yticklabels=herbs_sequence,
        annot=False,
    )
    plt.title("Accumulated Attention")
    plt.xlabel("Herbs")
    plt.ylabel("Herbs")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


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
    compute_dataset_stats(preprocessed)
    combo_scores, spmi_by_context_len = compute_spmi_values(preprocessed)
    save_pmi_examples(combo_scores)
    plot_spmi_distribution(spmi_by_context_len)

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

    model.eval()
    example1 = ["숙지황", "산수유", "지골피"]
    example2 = ["감초", "건강"]
    run_sequential_example(
        model,
        device,
        preprocessed,
        example1,
        os.path.join("results", "tables", "example_recommendations_1.csv"),
        os.path.join("results", "figures", "figure_attention_matrix_1.png"),
    )
    run_sequential_example(
        model,
        device,
        preprocessed,
        example2,
        os.path.join("results", "tables", "example_recommendations_2.csv"),
        os.path.join("results", "figures", "figure_attention_matrix_2.png"),
    )

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
