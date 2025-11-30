import argparse
import os
from typing import List

import torch

from herbmind_data import PREPROCESSED_PATH, ensure_dirs, load_and_preprocess
from model import HerbMindModel

CHECKPOINT_PATH = os.path.join("checkpoints", "herbmind.pt")


def load_model(device):
    preprocessed = load_and_preprocess()
    model = HerbMindModel(num_herbs=len(preprocessed["herb2id"]))
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, preprocessed


def tokenize_input(text: str) -> List[str]:
    tokens = [t.strip() for t in text.replace("\n", " ").split(",") if t.strip()]
    if len(tokens) == 1 and " " in tokens[0]:
        tokens = [t for t in tokens[0].split(" ") if t]
    return tokens


def recommend_loop(device):
    model, preprocessed = load_model(device)
    herb2id = preprocessed["herb2id"]
    id2herb = preprocessed["id2herb"]

    print("HerbMind interactive recommender. Enter a comma-separated list of herbs (Korean names). Type 'exit' to quit.")
    while True:
        user_input = input("Herbs > ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        herbs = tokenize_input(user_input)
        unknown = [h for h in herbs if h not in herb2id]
        if unknown:
            print(f"Unknown herbs skipped: {', '.join(unknown)}")
        ids = [herb2id[h] for h in herbs if h in herb2id]
        if len(ids) < 1:
            print("Please provide at least one known herb.")
            continue

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.tensor([[1] * len(ids)], dtype=torch.float, device=device)
        with torch.no_grad():
            indices, values = model.recommend(input_ids, mask, topk=3)
        recs = [(id2herb[i.item()], values[0, idx].item()) for idx, i in enumerate(indices[0]) if i.item() not in ids]
        if not recs:
            print("No recommendations (all top predictions already present).")
            continue
        print("Top-3 suggestions:")
        for name, score in recs:
            print(f" - {name}: {score:.4f}")


def main():
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError("Checkpoint not found. Run main.py to train the model first.")
    recommend_loop(device)


if __name__ == "__main__":
    main()
