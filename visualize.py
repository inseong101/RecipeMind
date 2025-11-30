import importlib.util
from typing import Dict, Sequence

import numpy as np
from sklearn.manifold import TSNE


def _import_optional(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    return importlib.import_module(name)


plt = _import_optional("matplotlib.pyplot")
sns = _import_optional("seaborn")


def set_korean_font():
    if plt is None:
        return
    import matplotlib

    for font in ["AppleGothic", "NanumGothic"]:
        try:
            matplotlib.rcParams["font.family"] = font
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_length_heatmap(length_df, save_path: str):
    if plt is None or sns is None or length_df.empty:
        return
    import pandas as pd

    bins = [2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf]
    labels = ["2", "3", "4", "5", "6", "7", "8", "9", "10+"]
    length_df = length_df.copy()
    length_df["bin"] = pd.cut(length_df["length"], bins=bins, labels=labels, right=False)
    pivot = length_df.groupby("bin")["top1"].mean().reset_index()
    plt.figure(figsize=(8, 4))
    sns.heatmap(pivot.set_index("bin"), annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Accuracy by Prescription Length")
    plt.ylabel("Length Bin")
    plt.xlabel("Top-1 Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_tsne_embeddings(model, id2herb: dict, save_path: str):
    if plt is None:
        return
    embeddings = model.embed.weight.detach().cpu().numpy()[: len(id2herb)]
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
    plt.title("t-SNE of Herb Embeddings")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_attention_heatmap(attn: np.ndarray, rows: Sequence[str], cols: Sequence[str], save_path: str):
    if plt is None or sns is None:
        return
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn, cmap="OrRd", xticklabels=cols, yticklabels=rows)
    plt.xlabel("Context Herbs")
    plt.ylabel("Candidate Herbs")
    plt.title("Cross-Attention (Recommendation Steps)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pmi_heatmap(pmi: np.ndarray, id2herb: dict, herb_counts: Dict[str, int], save_path: str, top_k: int = 50):
    if plt is None or sns is None or pmi.size == 0:
        return
    sorted_herbs = sorted(herb_counts.items(), key=lambda kv: kv[1], reverse=True)
    selected = [name for name, _ in sorted_herbs[:top_k] if name in id2herb.values()]
    indices = [idx for idx, name in id2herb.items() if name in set(selected)]
    if not indices:
        return
    submatrix = pmi[np.ix_(indices, indices)]
    labels = [id2herb[i] for i in indices]
    plt.figure(figsize=(12, 10))
    sns.heatmap(submatrix, cmap="coolwarm", xticklabels=labels, yticklabels=labels, center=0)
    plt.title("PMI Heatmap (Top-frequency Herbs)")
    plt.xlabel("Herb")
    plt.ylabel("Herb")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_frequency_scatter(herb_df, save_path: str):
    if plt is None or sns is None or herb_df.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=herb_df, x="dataset_frequency", y="top1")
    plt.xscale("log")
    plt.xlabel("Herb frequency (log scale)")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Accuracy vs. Herb Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_overview_diagram(save_path: str):
    if plt is None:
        return
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.text(0.1, 0.8, "Dataset\n(KIOM CSV)", bbox=dict(boxstyle="round", facecolor="#d1e0ff"), fontsize=12)
    plt.arrow(0.23, 0.8, 0.2, 0, head_width=0.02, length_includes_head=True)
    plt.text(
        0.45,
        0.8,
        "Preprocess\n- Deduplicate\n- Build N-1 quizzes\n- Co-occurrence",
        bbox=dict(boxstyle="round", facecolor="#d1ffd1"),
        fontsize=12,
    )
    plt.arrow(0.62, 0.8, 0.2, 0, head_width=0.02, length_includes_head=True)
    plt.text(0.84, 0.8, "HerbMind\nModel", bbox=dict(boxstyle="round", facecolor="#ffe0cc"), fontsize=12)
    plt.text(0.84, 0.45, "Outputs\n- Metrics\n- Figures\n- Tables", bbox=dict(boxstyle="round", facecolor="#fff2cc"), fontsize=12)
    plt.arrow(0.86, 0.76, 0, -0.23, head_width=0.02, length_includes_head=True)
    plt.text(0.45, 0.35, "Visualization Suite\n(PMI, sPMI, Attention, t-SNE)", bbox=dict(boxstyle="round", facecolor="#e0d1ff"), fontsize=12)
    plt.arrow(0.65, 0.78, -0.15, -0.33, head_width=0.02, length_includes_head=True)
    plt.arrow(0.83, 0.75, -0.28, -0.32, head_width=0.02, length_includes_head=True)
    plt.title("HerbMind Turn-key Pipeline Overview")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_model_architecture(save_path: str):
    if plt is None:
        return
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.text(0.1, 0.6, "Input\nHerb IDs", bbox=dict(boxstyle="round", facecolor="#d1e0ff"), fontsize=12)
    plt.arrow(0.23, 0.6, 0.18, 0, head_width=0.02, length_includes_head=True)
    plt.text(0.43, 0.6, "Embedding\n+ Positional Mask", bbox=dict(boxstyle="round", facecolor="#d1ffd1"), fontsize=12)
    plt.arrow(0.57, 0.6, 0.18, 0, head_width=0.02, length_includes_head=True)
    plt.text(0.77, 0.65, "Set Attention Blocks\n(Self-Attn xN)", bbox=dict(boxstyle="round", facecolor="#ffe0cc"), fontsize=12)
    plt.arrow(0.86, 0.6, 0, -0.2, head_width=0.02, length_includes_head=True)
    plt.text(0.77, 0.3, "Cross-Attention\nCandidates vs Context", bbox=dict(boxstyle="round", facecolor="#fff2cc"), fontsize=12)
    plt.arrow(0.7, 0.32, -0.2, 0, head_width=0.02, length_includes_head=True)
    plt.text(0.46, 0.3, "MLP Projection\n(Score each herb)", bbox=dict(boxstyle="round", facecolor="#e0d1ff"), fontsize=12)
    plt.arrow(0.35, 0.32, -0.15, 0, head_width=0.02, length_includes_head=True)
    plt.text(0.15, 0.3, "Top-k\nRecommendations", bbox=dict(boxstyle="round", facecolor="#d1e0ff"), fontsize=12)
    plt.title("HerbMind Architecture")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
