"""简化版电商 NER 训练栈的模型定义。

这个模块实现了一个最小可用的 ``Encoder + GlobalPointer`` 架构，能够直接
适配当前 HuggingFace 生态中的中文编码器模型。

当前提供的函数和类：
- ``rotate_half``：旋转位置编码中的辅助函数。
- ``apply_rope``：在 span 打分前为 query 和 key 应用 RoPE。
- ``GlobalPointer``：根据编码后的序列表示计算每个标签下的 span 分数。
- ``GlobalPointerForNer``：把 ``AutoModel`` 编码器和 GlobalPointer 头
  封装成统一训练/推理模型。
- ``global_pointer_loss``：微调阶段使用的多标签 span 损失函数。

这个文件没有 ``main`` 函数，它是一个可复用的模型模块，供训练和预测脚本
导入使用。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    seq_len = x.size(1)
    dim = x.size(-1)
    position_ids = torch.arange(seq_len, device=x.device).float()
    indices = torch.arange(0, dim, 2, device=x.device).float()
    inv_freq = torch.pow(10000.0, -indices / dim)
    sinusoid = torch.einsum("n,d->nd", position_ids, inv_freq)
    sin = torch.repeat_interleave(sinusoid.sin(), 2, dim=-1)[None, :, None, :]
    cos = torch.repeat_interleave(sinusoid.cos(), 2, dim=-1)[None, :, None, :]
    return x * cos + rotate_half(x) * sin


class GlobalPointer(nn.Module):
    def __init__(self, hidden_size: int, heads: int, head_size: int, rope: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.rope = rope
        self.proj = nn.Linear(hidden_size, heads * head_size * 2)

    def forward(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = sequence_output.shape
        x = self.proj(sequence_output).view(batch_size, seq_len, self.heads, self.head_size * 2)
        qw, kw = x[..., : self.head_size], x[..., self.head_size :]
        if self.rope:
            qw = apply_rope(qw)
            kw = apply_rope(kw)
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw) / math.sqrt(self.head_size)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
            logits = logits.masked_fill(~mask, -1e12)
            logits = logits.masked_fill(~mask.transpose(-1, -2), -1e12)
        lower_triangle = torch.tril(torch.ones(seq_len, seq_len, device=logits.device, dtype=torch.bool), diagonal=-1)
        logits = logits.masked_fill(lower_triangle[None, None, :, :], -1e12)
        return logits


class GlobalPointerForNer(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        head_size: int = 64,
        dropout: float = 0.1,
        rope: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.encoder_config)
        self.dropout = nn.Dropout(dropout)
        self.pointer = GlobalPointer(
            hidden_size=self.encoder_config.hidden_size,
            heads=num_labels,
            head_size=head_size,
            rope=rope,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        return self.pointer(sequence_output, attention_mask=attention_mask)


def global_pointer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    batch_size, num_labels = logits.shape[:2]
    y_pred = logits.reshape(batch_size * num_labels, -1)
    y_true = labels.reshape(batch_size * num_labels, -1)
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()
