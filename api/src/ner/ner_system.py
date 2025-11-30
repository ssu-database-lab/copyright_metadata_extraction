# ner_system.py
"""
배포/서비스에서 사용할 간단한 NER 추론 유틸.

예시:
    from ner_system import load_model_for_inference, predict_texts

    model, tokenizer, id2label, config, device = load_model_for_inference("saved_models/ner_bert_bilstm_crf")
    results = predict_texts(model, tokenizer, id2label, config, ["문장1", "문장2"], device=device)
"""

import argparse
from typing import List, Dict, Tuple

import torch

try:
    from .ner_model import load_ner_model
except ImportError:
    from ner_model import load_ner_model


def load_model_for_inference(
    model_dir: str,
    device: torch.device = None,
):
    """
    추론용으로 모델을 로드하는 헬퍼.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, label2id, id2label, config = load_ner_model(model_dir, device)
    return model, tokenizer, id2label, config, device


def predict_texts(
    model,
    tokenizer,
    id2label: Dict[int, str],
    config,
    texts: List[str],
    device: torch.device,
) -> List[Dict]:
    """
    raw 텍스트 리스트에 대해 NER 결과를 반환.

    반환 형식:
        [
            {
                "text": 원문 문자열,
                "tokens": [(token_str, label_str), ...]
            },
            ...
        ]
    """
    if isinstance(texts, str):
        texts = [texts]

    # BERT 토크나이저로 바로 인코딩
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    model.eval()
    results = []

    with torch.no_grad():
        pred_paths = model(input_ids=input_ids, attention_mask=attention_mask)

        for i, text in enumerate(texts):
            seq_len = int(attention_mask[i].sum().item())
            token_ids = input_ids[i][:seq_len]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            preds = pred_paths[i]
            if len(preds) != len(tokens):
                length = min(len(preds), len(tokens))
                preds = preds[:length]
                tokens = tokens[:length]
                token_ids = token_ids[:length]

            token_labels: List[Tuple[str, str]] = []
            for j, (token, label_id) in enumerate(zip(tokens, preds)):
                # CLS, SEP 등 special token은 ID로 확인하여 스킵
                if token_ids[j].item() in tokenizer.all_special_ids:
                    continue
                label_str = id2label[label_id]
                token_labels.append((token, label_str))

            results.append(
                {
                    "text": text,
                    "tokens": token_labels,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run NER inference on text")
    parser.add_argument("--model_dir", type=str, required=True, help="학습된 모델 디렉토리")
    parser.add_argument("--text", type=str, default=None, help="단일 입력 문장")
    parser.add_argument("--input_file", type=str, default=None, help="문장 하나당 한 줄씩 있는 파일")
    args = parser.parse_args()

    if not args.text and not args.input_file:
        print("Either --text or --input_file must be provided.")
        return

    model, tokenizer, id2label, config, device = load_model_for_inference(args.model_dir)

    if args.text:
        texts = [args.text]
    else:
        with open(args.input_file, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    results = predict_texts(model, tokenizer, id2label, config, texts, device)

    for res in results:
        print("TEXT:", res["text"])
        for token, label in res["tokens"]:
            print(f"{token}\t{label}")
        print("=" * 40)


if __name__ == "__main__":
    main()