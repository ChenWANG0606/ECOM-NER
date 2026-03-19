"""基于 span 的 NER 输出解码与评估辅助工具。

这个模块把指标计算逻辑从训练脚本中拆出来，使同一套解码规则可以同时复用
在验证和推理阶段。

当前提供的内容：
- ``Span``：标准 span 元组形式 ``(label_id, start_token, end_token)``。
- ``decode_global_pointer``：把模型 logits 解码为预测 span 集合，并过滤
  padding 区域的无效位置。
- ``compute_prf``：计算 span 级 precision、recall 和 F1。

这个文件没有 ``main`` 函数，主要被训练和推理入口脚本导入。
"""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

import torch


Span = Tuple[int, int, int]


def decode_global_pointer(
    logits: torch.Tensor,
    valid_token_mask: torch.Tensor,
    threshold: float = 0.0,
) -> List[Set[Span]]:
    predictions: List[Set[Span]] = []
    batch_size, num_labels, seq_len, _ = logits.shape
    for batch_idx in range(batch_size):
        valid_positions = valid_token_mask[batch_idx].bool()
        sample: Set[Span] = set()
        for label_id in range(num_labels):
            active = torch.nonzero(logits[batch_idx, label_id] > threshold, as_tuple=False)
            for start, end in active.tolist():
                if start > end:
                    continue
                if start >= seq_len or end >= seq_len:
                    continue
                if not valid_positions[start] or not valid_positions[end]:
                    continue
                sample.add((label_id, start, end))
        predictions.append(sample)
    return predictions


def compute_prf(predictions: Sequence[Set[Span]], references: Sequence[Set[Span]]) -> dict:
    true_positive = 0
    predicted_total = 0
    reference_total = 0
    for pred, ref in zip(predictions, references):
        true_positive += len(pred & ref)
        predicted_total += len(pred)
        reference_total += len(ref)
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / reference_total if reference_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "predicted": predicted_total,
        "gold": reference_total,
        "matched": true_positive,
    }
