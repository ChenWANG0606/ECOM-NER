#!/usr/bin/env python3
"""基于 build_corpus.py 生成的纯文本语料执行 Masked LM 继续预训练。

当前提供的函数和类：
- ``parse_args``：读取继续预训练配置文件路径。
- ``load_config``：把 JSON 配置加载到内存。
- ``build_autocast_context``：根据设备和精度选择混合精度上下文。
- ``load_corpus_lines``：读取一行一个标题的语料文件。
- ``split_corpus``：按照给定比例切分训练/验证文本。
- ``LineByLineMlmDataset``：把标题文本编码成 MLM 训练样本。
- ``evaluate``：在验证集上计算平均 MLM loss。
- ``save_checkpoint``：保存最佳 MLM 权重、encoder 权重和 tokenizer。
- ``main``：完成 tokenizer、数据集、模型、优化器、训练循环和模型导出。

``main`` 的执行思路：
1. 加载配置并创建输出目录。
2. 读取由 ``build_corpus.py`` 生成的一行一个标题的语料文件。
3. 切分训练/验证集并编码为动态 masking 所需样本。
4. 使用 ``AutoModelForMaskedLM`` 执行继续预训练。
5. 保存最佳 MLM checkpoint 以及可直接给 NER 微调使用的 encoder 目录。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from ecom_ner.train_utils import save_json, set_seed


class LineByLineMlmDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int) -> None:
        self.features = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )

    def __len__(self) -> int:
        return len(self.features["input_ids"])

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {name: values[index] for name, values in self.features.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue pretraining a masked language model on plain-text corpus.")
    parser.add_argument("--config", type=Path, default=Path("scripts/configs/continue_pretrain_mlm.json"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_autocast_context(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def load_corpus_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def split_corpus(lines: list[str], dev_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    dev_size = int(len(shuffled) * dev_ratio)
    if dev_size <= 0:
        return shuffled, []
    return shuffled[dev_size:], shuffled[:dev_size]


def resolve_encoder_module(model: AutoModelForMaskedLM):
    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return base_model
    prefix = getattr(model, "base_model_prefix", "")
    return getattr(model, prefix, model)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


@torch.no_grad()
def evaluate(model, dataloader, device, precision: str) -> dict:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with build_autocast_context(precision, device):
            outputs = model(**batch)
        total_loss += outputs.loss.item()
        total_steps += 1
    return {"mlm_loss": round(total_loss / max(total_steps, 1), 6)}


def save_checkpoint(model, tokenizer, output_dir: Path, metrics: dict) -> None:
    model_dir = output_dir / "best_model"
    encoder_dir = output_dir / "best_encoder"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    resolve_encoder_module(model).save_pretrained(encoder_dir)
    tokenizer.save_pretrained(encoder_dir)
    save_json(output_dir / "best_metrics.json", metrics)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = config.get("precision", "fp32")

    corpus_file = Path(config["corpus_file"])
    lines = load_corpus_lines(corpus_file)
    if not lines:
        raise ValueError(f"No non-empty lines found in corpus file: {corpus_file}")

    dev_ratio = float(config.get("dev_ratio", 0.0))
    train_texts, dev_texts = split_corpus(lines, dev_ratio=dev_ratio, seed=seed)
    if not train_texts:
        raise ValueError("Training corpus is empty after split. Reduce dev_ratio or provide more texts.")

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_name_or_path", config["model_name_or_path"]),
        use_fast=True,
    )
    max_length = int(config.get("max_length", 128))
    train_dataset = LineByLineMlmDataset(train_texts, tokenizer=tokenizer, max_length=max_length)
    dev_dataset = LineByLineMlmDataset(dev_texts, tokenizer=tokenizer, max_length=max_length) if dev_texts else None
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(config.get("mlm_probability", 0.15)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collator,
    )
    dev_loader = None
    if dev_dataset is not None:
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=int(config.get("eval_batch_size", config["train_batch_size"])),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=collator,
        )

    model = AutoModelForMaskedLM.from_pretrained(config["model_name_or_path"])
    if config.get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)

    no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight")
    optimizer_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(key in n for key in no_decay)],
            "weight_decay": float(config.get("weight_decay", 0.01)),
        },
        {
            "params": [p for n, p in model.named_parameters() if any(key in n for key in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_groups, lr=float(config["learning_rate"]))

    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    total_training_steps = math.ceil(len(train_loader) / grad_accum_steps) * int(config["num_epochs"])
    warmup_steps = int(total_training_steps * float(config.get("warmup_ratio", 0.1)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision == "fp16")

    save_json(output_dir / "train_config.json", config)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    dataset_summary = {
        "corpus_file": str(corpus_file),
        "train_texts": len(train_texts),
        "dev_texts": len(dev_texts),
        "train_batches": len(train_loader),
        "dev_batches": len(dev_loader) if dev_loader is not None else 0,
    }
    save_json(output_dir / "dataset_summary.json", dataset_summary)

    best_metric = float("inf")
    log_rows = []
    global_step = 0

    for epoch in range(1, int(config["num_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        running_loss = 0.0

        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            with build_autocast_context(precision, device):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item() * grad_accum_steps

            if step % grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(loss=f"{running_loss / step:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if len(train_loader) % grad_accum_steps != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        metrics = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(len(train_loader), 1), 6),
            "global_step": global_step,
        }
        if dev_loader is not None:
            metrics.update(evaluate(model=model, dataloader=dev_loader, device=device, precision=precision))
            monitor_value = metrics["mlm_loss"]
        else:
            monitor_value = metrics["train_loss"]
            metrics["mlm_loss"] = metrics["train_loss"]

        log_rows.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        if monitor_value < best_metric:
            best_metric = monitor_value
            save_checkpoint(model=model, tokenizer=tokenizer, output_dir=output_dir, metrics=metrics)

    save_json(output_dir / "training_log.json", {"history": log_rows})
    print(f"Continue pretraining completed. Best MLM loss: {best_metric:.6f}")


if __name__ == "__main__":
    main()
