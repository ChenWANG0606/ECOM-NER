#!/usr/bin/env python3
"""使用训练好的 GlobalPointer NER checkpoint 执行推理。

当前提供的函数：
- ``parse_args``：定义 checkpoint 路径、输入路径、输出路径、batch size 和
  解码阈值。
- ``load_examples``：接受准备好的 JSONL 或纯文本标题，并转换成统一样本
  结构。
- ``main``：加载模型产物，构建预测 dataloader，完成解码并写出 JSONL
  预测结果。

``main`` 的执行思路：
1. 从 checkpoint 目录读取训练配置和标签列表。
2. 如果输入是纯文本，则先临时转换成 JSONL。
3. 用保存下来的产物重建 tokenizer、数据集、dataloader 和模型。
4. 在 CPU 或 GPU 上执行批量推理。
5. 把 token 级 span 解码回字符级实体区间。
6. 保存预测结果，并在需要时删除临时文件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ecom_ner.data import GlobalPointerCollator, JsonlNERDataset
from ecom_ner.io import read_jsonl, write_jsonl
from ecom_ner.labels import label_to_id_map
from ecom_ner.metrics import decode_global_pointer
from ecom_ner.modeling import GlobalPointerForNer
from ecom_ner.train_utils import to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict NER spans with a trained GlobalPointer model.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


def load_examples(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            text = line.rstrip("\n").rstrip("\r")
            if not text:
                continue
            rows.append({"id": f"raw-{idx}", "text": text, "tokens": list(text), "entities": [], "source": "raw"})
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads((args.checkpoint_dir / "train_config.json").read_text(encoding="utf-8"))
    labels = json.loads((args.checkpoint_dir / "labels.json").read_text(encoding="utf-8"))["labels"]
    id_to_label = {idx: label for idx, label in enumerate(labels)}
    label_to_id = label_to_id_map(labels)

    temp_input = args.input_file
    created_temp_file = False
    if temp_input.suffix != ".jsonl":
        examples = load_examples(temp_input)
        temp_input = args.checkpoint_dir / "_predict_input.jsonl"
        write_jsonl(temp_input, examples)
        created_temp_file = True

    tokenizer_dir = args.checkpoint_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    dataset = JsonlNERDataset(
        path=temp_input,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=int(config["max_length"]),
    )
    collator = GlobalPointerCollator(tokenizer=tokenizer, num_labels=len(labels))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = GlobalPointerForNer(
        model_name_or_path=config["model_name_or_path"],
        num_labels=len(labels),
        head_size=int(config.get("head_size", 64)),
        dropout=float(config.get("dropout", 0.1)),
        rope=bool(config.get("rope", True)),
    )
    model.load_state_dict(torch.load(args.checkpoint_dir / "best_model.pt", map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="predict", leave=False):
            batch = to_device(batch, device)
            logits = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                token_type_ids=batch.token_type_ids,
            )
            span_sets = decode_global_pointer(logits.float().cpu(), batch.valid_token_mask.cpu(), args.threshold)
            for sample_idx, spans in enumerate(span_sets):
                text = batch.texts[sample_idx]
                word_ids = batch.word_ids[sample_idx]
                entities = []
                for label_id, start_token, end_token in sorted(spans, key=lambda x: (x[1], x[2], x[0])):
                    start_char = word_ids[start_token]
                    end_char = word_ids[end_token] + 1
                    if start_char < 0 or end_char <= start_char:
                        continue
                    entities.append(
                        {
                            "start": start_char,
                            "end": end_char,
                            "label": id_to_label[label_id],
                            "text": text[start_char:end_char],
                            "score": round(float(logits[sample_idx, label_id, start_token, end_token].item()), 6),
                        }
                    )
                predictions.append(
                    {
                        "id": batch.ids[sample_idx],
                        "text": text,
                        "entities": entities,
                    }
                )

    write_jsonl(args.output_file, predictions)
    if created_temp_file and temp_input.exists():
        temp_input.unlink()
    print(f"Saved predictions to {args.output_file}")


if __name__ == "__main__":
    main()
