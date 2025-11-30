import argparse
import os
from typing import List

import torch

from herbmind_data import ensure_dirs, load_and_preprocess
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
    tokens = [t.strip() for t in text.replace("\n", " ").replace("/", ",").split(",") if t.strip()]
    if len(tokens) == 1 and " " in tokens[0]:
        tokens = [t for t in tokens[0].split(" ") if t]
    return tokens


def recommend_loop(device):
    model, preprocessed = load_model(device)
    herb2id = preprocessed["herb2id"]
    id2herb = preprocessed["id2herb"]

    current: List[str] = []
    print("HerbMind interactive recommender. 입력한 1~2개 약재를 기반으로 Top-3 약재를 제안합니다. 'done' 또는 'exit' 입력 시 종료.")
    while True:
        print(f"현재 처방: {', '.join(current) if current else '비어 있음'}")
        user_input = input("추가할 약재 (쉼표 또는 공백 구분) > ").strip()
        if user_input.lower() in {"exit", "quit", "done"}:
            print("최종 처방:", ", ".join(current))
            break

        herbs = tokenize_input(user_input)
        if not herbs:
            continue

        unknown = [h for h in herbs if h not in herb2id]
        if unknown:
            print(f"알 수 없는 약재가 제외되었습니다: {', '.join(unknown)}")
        for h in herbs:
            if h in herb2id and h not in current:
                current.append(h)

        ids = [herb2id[h] for h in current if h in herb2id]
        if len(ids) < 1:
            print("알려진 약재를 하나 이상 입력해 주세요.")
            continue

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.tensor([[1] * len(ids)], dtype=torch.float, device=device)
        with torch.no_grad():
            indices, values = model.recommend(input_ids, mask, topk=3)
        recs = []
        for idx, i in enumerate(indices[0]):
            herb_id = i.item()
            name = id2herb.get(herb_id, str(herb_id))
            if name in current:
                continue
            recs.append((name, values[0, idx].item()))
        if not recs:
            print("모든 추천이 이미 처방에 포함되어 있습니다.")
            continue
        print("추천 약재 Top-3:")
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
