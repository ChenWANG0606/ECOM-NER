"""合并版 NER 项目的输入输出与数据格式转换工具。

这个模块位于原始语料文件和项目统一 JSONL 中间格式之间，负责把不同来源、
不同格式的数据整理成同一套样本结构。

当前提供的函数：
- ``ensure_dir``：安全创建输出目录。
- ``read_jsonl`` / ``write_jsonl``：读取和写入项目统一 JSONL 文件。
- ``read_conll_sentences``：读取 token-label 格式语料，兼容空格 token 和
  特殊字符。
- ``read_word_per_line_titles``：读取 GAIIC 风格的逐词一行无标注标题。
- ``read_line_titles``：读取一行一个完整标题的文本语料。
- ``bio_tags_to_entities``：把 BIO 序列转换成 span 字典列表。
- ``build_example``：把 tokens 和实体信息打包成统一样本结构。
- ``load_labeled_conll``：把有标注源文件转换成可直接写入 JSONL 的样本。
- ``load_unlabeled_word_per_line`` / ``load_unlabeled_lines``：把无标注
  语料也转换成相同样本结构。
- ``summarize_examples``：统计样本数、实体数、长度和标签分布。
- ``dump_json``：写出统计信息或配置类 JSON 文件。

这个文件没有 ``main`` 函数，主要被数据准备、语料构建以及训练前处理脚本
复用。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from .labels import normalize_bio_tag


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_conll_sentences(path: Path) -> Iterator[List[tuple[str, str]]]:
    sentence: List[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n").rstrip("\r")
            if raw == "":
                if sentence:
                    yield sentence
                    sentence = []
                continue
            if len(raw) >= 3 and raw[1] == " ":
                token = raw[0]
                label = raw[2:]
            else:
                parts = raw.split(maxsplit=1)
                if len(parts) == 1:
                    token = parts[0]
                    label = "O"
                else:
                    token, label = parts[0], parts[1]
            sentence.append((token, label))
    if sentence:
        yield sentence


def read_word_per_line_titles(path: Path) -> Iterator[List[str]]:
    tokens: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n").rstrip("\r")
            if raw == "":
                if tokens:
                    yield tokens
                    tokens = []
                continue
            tokens.append(raw)
    if tokens:
        yield tokens


def read_line_titles(path: Path) -> Iterator[List[str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n").rstrip("\r")
            if text:
                yield list(text)


def bio_tags_to_entities(tags: Sequence[str]) -> List[Dict]:
    entities: List[Dict] = []
    start = -1
    label_name = ""
    for idx, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            if start >= 0:
                entities.append({"start": start, "end": idx, "label": label_name})
                start = -1
                label_name = ""
            continue
        prefix, current_label = tag.split("-", 1)
        if prefix == "B" or current_label != label_name:
            if start >= 0:
                entities.append({"start": start, "end": idx, "label": label_name})
            start = idx
            label_name = current_label
        elif prefix != "I":
            raise ValueError(f"Invalid BIO tag: {tag}")
    return entities


def build_example(example_id: str, tokens: Sequence[str], entities: List[Dict], source: str) -> Dict:
    text = "".join(tokens)
    resolved = []
    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        resolved.append(
            {
                "start": start,
                "end": end,
                "label": entity["label"],
                "text": text[start:end],
            }
        )
    return {
        "id": example_id,
        "text": text,
        "tokens": list(tokens),
        "entities": resolved,
        "source": source,
    }


def load_labeled_conll(path: Path, source: str, prefix: str) -> List[Dict]:
    rows: List[Dict] = []
    for idx, sentence in enumerate(read_conll_sentences(path)):
        tokens = [token for token, _ in sentence]
        tags = [normalize_bio_tag(tag, source) for _, tag in sentence]
        entities = bio_tags_to_entities(tags)
        rows.append(build_example(f"{prefix}-{idx}", tokens, entities, source))
    return rows


def load_unlabeled_word_per_line(path: Path, prefix: str, source: str) -> List[Dict]:
    return [build_example(f"{prefix}-{idx}", tokens, [], source) for idx, tokens in enumerate(read_word_per_line_titles(path))]


def load_unlabeled_lines(path: Path, prefix: str, source: str) -> List[Dict]:
    return [build_example(f"{prefix}-{idx}", tokens, [], source) for idx, tokens in enumerate(read_line_titles(path))]


def summarize_examples(examples: Sequence[Dict]) -> Dict:
    entity_counter: Counter[str] = Counter()
    entity_count = 0
    lengths: List[int] = []
    for example in examples:
        lengths.append(len(example["tokens"]))
        for entity in example["entities"]:
            entity_counter[entity["label"]] += 1
            entity_count += 1
    avg_len = round(sum(lengths) / len(lengths), 2) if lengths else 0.0
    return {
        "samples": len(examples),
        "entities": entity_count,
        "avg_length": avg_len,
        "label_distribution": dict(sorted(entity_counter.items())),
    }


def dump_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
