#!/usr/bin/env python3
"""从准备好的 JSONL 数据集中构建纯文本语料文件。

当前提供的函数：
- ``parse_args``：定义输入数据文件列表和输出语料路径。
- ``main``：从 JSONL 或原始文本中提取标题，最终写成一行一个标题的语料
  文件。

``main`` 的执行思路：
1. 读取命令行参数。
2. 遍历每个输入文件，并跳过不存在的路径。
3. 从准备好的 JSONL 或逐行文本语料中提取标题文本。
4. 拼接成统一纯文本语料文件，供继续预训练或语料统计使用。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecom_ner.io import load_unlabeled_lines, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a plain-text corpus from prepared JSONL files.")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            Path("data/processed/merged_ner/train_merged.jsonl"),
            Path("data/processed/merged_ner/dev.jsonl"),
            Path("data/processed/merged_ner/unlabeled.jsonl"),
        ],
    )
    parser.add_argument("--output-file", type=Path, default=Path("data/processed/merged_ner/corpus.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    titles = []
    for path in args.inputs:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            titles.extend(example["text"] for example in read_jsonl(path))
        else:
            titles.extend("".join(example["tokens"]) for example in load_unlabeled_lines(path, "tmp", "tmp"))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text("\n".join(titles) + "\n", encoding="utf-8")
    print(f"Saved corpus to {args.output_file} with {len(titles)} titles")


if __name__ == "__main__":
    main()
