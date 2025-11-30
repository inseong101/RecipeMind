import importlib.util
from typing import Sequence

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


def plot_pmi_heatmap(pmi: np.ndarray, id2herb: dict, save_path: str, top_k: int = 20):
    if plt is None or sns is None or pmi.size == 0:
        return
    names = [id2herb[i] for i in range(len(id2herb))]
    abs_scores = np.abs(pmi)
    top_indices = np.unravel_index(np.argsort(abs_scores, axis=None)[-top_k:], abs_scores.shape)
    mask = np.zeros_like(pmi, dtype=bool)
    mask[top_indices] = True
    plt.figure(figsize=(10, 8))
    sns.heatmap(pmi, cmap="coolwarm", xticklabels=names, yticklabels=names, mask=~mask)
    plt.title("Pointwise Mutual Information of Herb Pairs")
    plt.xlabel("Herb")
    plt.ylabel("Herb")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
