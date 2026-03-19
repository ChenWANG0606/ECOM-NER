#!/usr/bin/env python3
"""读取训练好的 GlobalPointer checkpoint，并按实体类型评估验证集指标。

当前提供的函数：
- ``parse_args``：定义 checkpoint、数据文件、输出路径和解码阈值参数。
- ``load_runtime_config``：从 checkpoint 目录恢复训练配置、标签和 tokenizer。
- ``compute_span_prf``：计算单个实体类型的 precision、recall、F1。
- ``compute_auc_by_label``：基于 span logits 计算每个实体类型的 ROC AUC。
- ``evaluate``：执行完整验证流程并汇总 overall + per-label 指标。
- ``main``：加载模型和验证集，运行评估并保存结果。

这个脚本用于离线分析模型在验证集上的不同实体类型表现，尤其适合比较哪些类
的召回低、哪些类的打分可分性差。
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ecom_ner.data import GlobalPointerCollator, JsonlNERDataset
from ecom_ner.labels import label_to_id_map
from ecom_ner.metrics import compute_prf, decode_global_pointer
from ecom_ner.modeling import GlobalPointerForNer
from ecom_ner.train_utils import to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a GlobalPointer NER checkpoint by entity type.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, default=None, help="Defaults to dev_file in train_config.json.")
    parser.add_argument("--output-file", type=Path, default=None, help="Defaults to checkpoint_dir/eval_by_label.json.")
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to eval_batch_size in train_config.json.")
    parser.add_argument("--threshold", type=float, default=None, help="Defaults to eval_threshold in train_config.json.")
    return parser.parse_args()


def load_runtime_config(checkpoint_dir: Path) -> tuple[dict, list[str], AutoTokenizer]:
    config = json.loads((checkpoint_dir / "train_config.json").read_text(encoding="utf-8"))
    labels = json.loads((checkpoint_dir / "labels.json").read_text(encoding="utf-8"))["labels"]
    tokenizer_dir = checkpoint_dir / "tokenizer"
    tokenizer_source = tokenizer_dir if tokenizer_dir.exists() else config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    return config, labels, tokenizer


def compute_span_prf(predictions: list[set[tuple[int, int, int]]], references: list[set[tuple[int, int, int]]]) -> dict:
    metrics = compute_prf(predictions, references)
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "predicted": metrics["predicted"],
        "gold": metrics["gold"],
        "matched": metrics["matched"],
    }


def build_valid_span_mask(valid_token_mask: torch.Tensor) -> torch.Tensor:
    seq_len = valid_token_mask.size(0)
    valid_positions = valid_token_mask.bool()
    upper_triangle = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=valid_token_mask.device))
    return valid_positions[:, None] & valid_positions[None, :] & upper_triangle


def compute_auc_by_label(
    logits_batches: list[torch.Tensor],
    label_batches: list[torch.Tensor],
    valid_token_masks: list[torch.Tensor],
    labels: list[str],
) -> dict[str, float | None]:
    score_store = {label: [] for label in labels}
    target_store = {label: [] for label in labels}

    for logits, gold_labels, valid_mask in zip(logits_batches, label_batches, valid_token_masks):
        batch_size, num_labels, _, _ = logits.shape
        for batch_idx in range(batch_size):
            span_mask = build_valid_span_mask(valid_mask[batch_idx])
            for label_id in range(num_labels):
                score_store[labels[label_id]].append(logits[batch_idx, label_id][span_mask].reshape(-1))
                target_store[labels[label_id]].append(gold_labels[batch_idx, label_id][span_mask].reshape(-1))

    auc_by_label: dict[str, float | None] = {}
    for label in labels:
        scores = torch.cat(score_store[label]).numpy()
        targets = torch.cat(target_store[label]).numpy()
        if len(set(targets.tolist())) < 2:
            auc_by_label[label] = None
            continue
        auc_by_label[label] = round(float(roc_auc_score(targets, scores)), 6)
    return auc_by_label


@torch.no_grad()
def evaluate(model, dataloader, device, labels: list[str], threshold: float) -> dict:
    model.eval()
    predictions = []
    golds = []
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    valid_token_masks: list[torch.Tensor] = []

    for batch in tqdm(dataloader, desc="evaluate", leave=False):
        batch = to_device(batch, device)
        logits = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            token_type_ids=batch.token_type_ids,
        )
        logits_cpu = logits.float().cpu()
        labels_cpu = batch.labels.float().cpu()
        valid_mask_cpu = batch.valid_token_mask.cpu()

        predictions.extend(decode_global_pointer(logits_cpu, valid_mask_cpu, threshold))
        golds.extend(batch.gold_spans)
        logits_batches.append(logits_cpu)
        label_batches.append(labels_cpu)
        valid_token_masks.append(valid_mask_cpu)

    overall_metrics = compute_span_prf(predictions, golds)
    auc_by_label = compute_auc_by_label(logits_batches, label_batches, valid_token_masks, labels)

    per_label = {}
    for label_id, label_name in enumerate(labels):
        label_predictions = [{span for span in sample if span[0] == label_id} for sample in predictions]
        label_golds = [{span for span in sample if span[0] == label_id} for sample in golds]
        metrics = compute_span_prf(label_predictions, label_golds)
        metrics["auc"] = auc_by_label[label_name]
        per_label[label_name] = metrics

    valid_auc_values = [value for value in auc_by_label.values() if value is not None]
    overall_metrics["macro_auc"] = round(sum(valid_auc_values) / len(valid_auc_values), 6) if valid_auc_values else None
    return {
        "overall": overall_metrics,
        "per_label": per_label,
    }


def main() -> None:
    args = parse_args()
    config, labels, tokenizer = load_runtime_config(args.checkpoint_dir)
    label_to_id = label_to_id_map(labels)

    input_file = args.input_file or Path(config["dev_file"])
    batch_size = args.batch_size or int(config.get("eval_batch_size", 32))
    threshold = args.threshold if args.threshold is not None else float(config.get("eval_threshold", 0.0))
    output_file = args.output_file or (args.checkpoint_dir / "eval_by_label.json")

    dataset = JsonlNERDataset(
        path=input_file,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=int(config["max_length"]),
    )
    collator = GlobalPointerCollator(tokenizer=tokenizer, num_labels=len(labels))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collator,
    )

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

    results = evaluate(model=model, dataloader=dataloader, device=device, labels=labels, threshold=threshold)
    results["meta"] = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "input_file": str(input_file),
        "batch_size": batch_size,
        "threshold": threshold,
    }
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(results["overall"], ensure_ascii=False))
    print(f"Saved per-label metrics to {output_file}")


if __name__ == "__main__":
    main()
