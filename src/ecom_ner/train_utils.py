"""项目脚本共享的小型训练辅助函数。

当前提供的函数：
- ``set_seed``：统一设置 Python、NumPy 和 PyTorch 的随机种子。
- ``to_device``：把拼装好的 batch 容器移动到目标设备上。
- ``save_json``：把结构化配置或指标信息写入磁盘。

这个文件没有 ``main`` 函数，它的作用是把通用小工具从入口脚本中拆出来，
让训练和预测逻辑更聚焦于主流程。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device: torch.device):
    batch.input_ids = batch.input_ids.to(device)
    batch.attention_mask = batch.attention_mask.to(device)
    batch.token_type_ids = batch.token_type_ids.to(device)
    batch.labels = batch.labels.to(device)
    batch.valid_token_mask = batch.valid_token_mask.to(device)
    return batch


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
