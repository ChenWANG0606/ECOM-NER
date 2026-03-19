"""GlobalPointer NER 训练所需的数据集与批处理拼装逻辑。

这个模块负责把准备好的 JSONL 样本转换成模型可用张量，并完成字符级实体
区间到分词后 token 区间的对齐。

当前提供的类：
- ``JsonlNERDataset``：读取准备好的 JSONL 数据，对每条样本分词，并把
  字符级实体区间转换为 token 级 span。
- ``GlobalPointerBatch``：定义一次完整 batch 的结构，包含训练张量以及评估
  和解码需要的元信息。
- ``GlobalPointerCollator``：对变长样本做 padding，并构造 GlobalPointer
  训练所需的稠密 span 标签张量。

这个文件没有 ``main`` 函数，主要由 ``scripts/train.py`` 和
``scripts/predict.py`` 直接调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from .io import read_jsonl


class JsonlNERDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        label_to_id: Dict[str, int],
        max_length: int,
    ) -> None:
        self.examples = read_jsonl(path)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        example = self.examples[index]
        tokens = example["tokens"]
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        word_ids = encoded.word_ids()
        first_token_for_word: Dict[int, int] = {}
        last_token_for_word: Dict[int, int] = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            first_token_for_word.setdefault(word_idx, token_idx)
            last_token_for_word[word_idx] = token_idx

        spans: List[tuple[int, int, int]] = []
        for entity in example["entities"]:
            start_word = entity["start"]
            end_word = entity["end"] - 1
            if start_word not in first_token_for_word or end_word not in last_token_for_word:
                continue
            spans.append(
                (
                    self.label_to_id[entity["label"]],
                    first_token_for_word[start_word],
                    last_token_for_word[end_word],
                )
            )

        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = [0] * len(encoded["input_ids"])

        return {
            "id": example["id"],
            "text": example["text"],
            "tokens": tokens,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "token_type_ids": token_type_ids,
            "word_ids": [-1 if item is None else int(item) for item in word_ids],
            "spans": spans,
        }


@dataclass
class GlobalPointerBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    labels: torch.Tensor
    valid_token_mask: torch.Tensor
    gold_spans: List[set[tuple[int, int, int]]]
    word_ids: List[List[int]]
    texts: List[str]
    ids: List[str]


class GlobalPointerCollator:
    def __init__(self, tokenizer, num_labels: int) -> None:
        self.tokenizer = tokenizer
        self.num_labels = num_labels

    def __call__(self, features: Sequence[Dict]) -> GlobalPointerBatch:
        batch_inputs = self.tokenizer.pad(
            [
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                    "token_type_ids": item["token_type_ids"],
                }
                for item in features
            ],
            padding=True,
            return_tensors="pt",
        )
        batch_size, seq_len = batch_inputs["input_ids"].shape
        labels = torch.zeros(batch_size, self.num_labels, seq_len, seq_len, dtype=torch.float32)
        valid_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        gold_spans: List[set[tuple[int, int, int]]] = []
        word_ids_batch: List[List[int]] = []
        texts: List[str] = []
        ids: List[str] = []

        for batch_idx, feature in enumerate(features):
            word_ids = feature["word_ids"] + [-1] * (seq_len - len(feature["word_ids"]))
            word_ids_batch.append(word_ids)
            ids.append(feature["id"])
            texts.append(feature["text"])
            gold_set: set[tuple[int, int, int]] = set()
            for label_id, start, end in feature["spans"]:
                if start >= seq_len or end >= seq_len or start > end:
                    continue
                labels[batch_idx, label_id, start, end] = 1.0
                gold_set.add((label_id, start, end))
            gold_spans.append(gold_set)
            valid_token_mask[batch_idx, : len(feature["word_ids"])] = torch.tensor(
                [item >= 0 for item in feature["word_ids"]],
                dtype=torch.bool,
            )

        return GlobalPointerBatch(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            token_type_ids=batch_inputs["token_type_ids"],
            labels=labels,
            valid_token_mask=valid_token_mask,
            gold_spans=gold_spans,
            word_ids=word_ids_batch,
            texts=texts,
            ids=ids,
        )
