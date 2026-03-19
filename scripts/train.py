#!/usr/bin/env python3
"""训练简化版 GlobalPointer NER 模型。

当前提供的函数：
- ``parse_args``：读取 JSON 训练配置文件路径。
- ``load_config``：把 JSON 配置加载到内存。
- ``build_autocast_context``：根据设备类型和精度配置选择混合精度上下文。
- ``evaluate``：执行验证集评估，完成解码并计算 PRF 和 loss。
- ``main``：统筹 tokenizer、数据集、模型、优化器、训练循环、验证和保存。

``main`` 的执行思路：
1. 加载配置并创建输出目录。
2. 设置随机种子，并选择 CPU 或 GPU。
3. 加载标签、tokenizer、准备好的 train/dev 数据集和 dataloader。
4. 构建编码器加 GlobalPointer 的模型，以及优化器和学习率调度器。
5. 按 epoch 执行训练，支持混合精度和梯度裁剪。
6. 每个 epoch 结束后在 dev 集上评估。
7. 保存最优 checkpoint、指标、tokenizer 和训练日志。
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ecom_ner.data import GlobalPointerCollator, JsonlNERDataset
from ecom_ner.labels import label_to_id_map
from ecom_ner.metrics import compute_prf, decode_global_pointer
from ecom_ner.modeling import GlobalPointerForNer, global_pointer_loss
from ecom_ner.train_utils import save_json, set_seed, to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GlobalPointer NER model.")
    parser.add_argument("--config", type=Path, default=Path("configs/global_pointer_baseline.json"))
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


@torch.no_grad()
def evaluate(model, dataloader, device, threshold: float, precision: str) -> dict:
    model.eval()
    predictions = []
    golds = []
    total_loss = 0.0
    total_steps = 0
    for batch in dataloader:
        batch = to_device(batch, device)
        with build_autocast_context(precision, device):
            logits = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                token_type_ids=batch.token_type_ids,
            )
            loss = global_pointer_loss(logits, batch.labels)
        total_loss += loss.item()
        total_steps += 1
        predictions.extend(decode_global_pointer(logits.float().cpu(), batch.valid_token_mask.cpu(), threshold))
        golds.extend(batch.gold_spans)
    metrics = compute_prf(predictions, golds)
    metrics["loss"] = round(total_loss / max(total_steps, 1), 6)
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = config.get("precision", "fp32")

    label_payload = json.loads(Path(config["label_file"]).read_text(encoding="utf-8"))
    labels = label_payload["labels"]
    label_to_id = label_to_id_map(labels)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True)
    train_dataset = JsonlNERDataset(
        path=Path(config["train_file"]),
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=int(config["max_length"]),
    )
    dev_dataset = JsonlNERDataset(
        path=Path(config["dev_file"]),
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=int(config["max_length"]),
    )
    collator = GlobalPointerCollator(tokenizer=tokenizer, num_labels=len(labels))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train_batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=int(config["eval_batch_size"]),
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
    if config.get("gradient_checkpointing", False) and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    model.to(device)

    encoder_params = list(model.encoder.named_parameters())
    pointer_params = [(name, param) for name, param in model.named_parameters() if not name.startswith("encoder.")]
    no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight")
    optimizer_groups = [
        {
            "params": [p for n, p in encoder_params if not any(key in n for key in no_decay)],
            "lr": float(config.get("encoder_learning_rate", config["learning_rate"])),
            "weight_decay": float(config.get("weight_decay", 0.01)),
        },
        {
            "params": [p for n, p in encoder_params if any(key in n for key in no_decay)],
            "lr": float(config.get("encoder_learning_rate", config["learning_rate"])),
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in pointer_params if not any(key in n for key in no_decay)],
            "lr": float(config.get("head_learning_rate", config["learning_rate"])),
            "weight_decay": float(config.get("weight_decay", 0.01)),
        },
        {
            "params": [p for n, p in pointer_params if any(key in n for key in no_decay)],
            "lr": float(config.get("head_learning_rate", config["learning_rate"])),
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_groups)

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
    save_json(output_dir / "labels.json", {"labels": labels})
    tokenizer.save_pretrained(output_dir / "tokenizer")

    best_f1 = -1.0
    log_rows = []
    global_step = 0

    for epoch in range(1, int(config["num_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        running_loss = 0.0
        for step, batch in enumerate(progress, start=1):
            batch = to_device(batch, device)
            with build_autocast_context(precision, device):
                logits = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    token_type_ids=batch.token_type_ids,
                )
                loss = global_pointer_loss(logits, batch.labels) / grad_accum_steps
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

        dev_metrics = evaluate(
            model=model,
            dataloader=dev_loader,
            device=device,
            threshold=float(config.get("eval_threshold", 0.0)),
            precision=precision,
        )
        row = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(len(train_loader), 1), 6),
            **dev_metrics,
        }
        log_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            save_json(output_dir / "best_metrics.json", row)

    save_json(output_dir / "training_log.json", {"history": log_rows})
    print(f"Training completed. Best F1: {best_f1:.6f}")


if __name__ == "__main__":
    main()
