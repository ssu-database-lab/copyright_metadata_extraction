# ner_evaluate.py
import argparse

import torch
from torch.utils.data import DataLoader

try:
    from .ner_model import load_ner_model
    from .ner_data import read_conll, NERDataset, evaluate_ner
except ImportError:
    from ner_model import load_ner_model
    from ner_data import read_conll, NERDataset, evaluate_ner


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NER model on test set")
    parser.add_argument("--model_dir", type=str, required=True, help="학습된 모델 디렉토리")
    parser.add_argument("--test_file", type=str, required=True, help="BIO test 파일 경로")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print(f"Loading model from {args.model_dir}")
    model, tokenizer, label2id, id2label, config = load_ner_model(args.model_dir, device)

    print(f"Loading test data from {args.test_file}")
    test_sentences = read_conll(args.test_file)

    test_dataset = NERDataset(test_sentences, tokenizer, label2id, max_length=config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    score, report = evaluate_ner(model, test_loader, id2label, device)
    print(f"\n[Test] score = {score:.4f}")
    print(report)


if __name__ == "__main__":
    main()