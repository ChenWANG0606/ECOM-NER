#!/usr/bin/env python3
"""准备新项目使用的合并版电商 NER 数据集。

当前提供的函数：
- ``parse_args``：定义命令行参数，包括原始数据根目录、输出目录、验证集比例
  和随机种子。
- ``split_train_dev``：对 GAIIC 标注样本做可复现的打乱，并切分 train/dev。
- ``main``：组织完整的数据预处理主流程。

``main`` 的执行思路：
1. 读取命令行参数，并确保处理后数据目录存在。
2. 加载并归一化 GAIIC 标注训练集。
3. 将 GAIIC 切分为训练子集和验证子集。
4. 加载外部 ``ecommerce`` 数据，并映射到统一标签空间。
5. 合并 GAIIC 训练集和 ``ecommerce`` 数据形成最终训练集。
6. 把 GAIIC 测试集和无标注语料转换成相同 JSONL 结构。
7. 写出 JSONL 数据、标签定义和统计信息。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecom_ner.io import (
    dump_json,
    ensure_dir,
    load_labeled_conll,
    load_unlabeled_lines,
    load_unlabeled_word_per_line,
    summarize_examples,
    write_jsonl,
)
from ecom_ner.labels import UNIFIED_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare merged e-commerce NER data.")
    parser.add_argument("--data-root", type=Path, default=Path("data/中文NER数据集"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/merged_ner"))
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_train_dev(rows: list[dict], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = rows[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    dev_size = max(1, int(len(shuffled) * dev_ratio))
    dev_rows = shuffled[:dev_size]
    train_rows = shuffled[dev_size:]
    return train_rows, dev_rows


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    gaiic_root = args.data_root / "商品标题2022-NER"
    ecommerce_root = args.data_root / "ecommerce"

    gaiic_rows = load_labeled_conll(gaiic_root / "train.txt", source="gaiic", prefix="gaiic")
    gaiic_train, gaiic_dev = split_train_dev(gaiic_rows, dev_ratio=args.dev_ratio, seed=args.seed)

    ecommerce_rows = []
    for split in ("train.txt", "dev.txt", "test.txt"):
        ecommerce_rows.extend(
            load_labeled_conll(ecommerce_root / split, source="ecommerce", prefix=f"ecommerce-{split[:-4]}")
        )

    merged_train = gaiic_train + ecommerce_rows
    test_a = load_unlabeled_word_per_line(
        gaiic_root / "preliminary_test_a/word_per_line_preliminary_A.txt",
        prefix="gaiic-test-a",
        source="gaiic-test-a",
    )
    test_b = load_unlabeled_word_per_line(
        gaiic_root / "preliminary_test_b/word_per_line_preliminary_B.txt",
        prefix="gaiic-test-b",
        source="gaiic-test-b",
    )
    unlabeled = load_unlabeled_lines(
        gaiic_root / "unlabeled_train_data.txt",
        prefix="gaiic-unlabeled",
        source="gaiic-unlabeled",
    )

    write_jsonl(args.output_dir / "train_gaiic.jsonl", gaiic_train)
    write_jsonl(args.output_dir / "dev.jsonl", gaiic_dev)
    write_jsonl(args.output_dir / "train_ecommerce.jsonl", ecommerce_rows)
    write_jsonl(args.output_dir / "train_merged.jsonl", merged_train)
    write_jsonl(args.output_dir / "test_a.jsonl", test_a)
    write_jsonl(args.output_dir / "test_b.jsonl", test_b)
    write_jsonl(args.output_dir / "unlabeled.jsonl", unlabeled)
    dump_json(args.output_dir / "labels.json", {"labels": UNIFIED_LABELS})
    dump_json(
        args.output_dir / "stats.json",
        {
            "train_gaiic": summarize_examples(gaiic_train),
            "dev": summarize_examples(gaiic_dev),
            "train_ecommerce": summarize_examples(ecommerce_rows),
            "train_merged": summarize_examples(merged_train),
            "test_a": summarize_examples(test_a),
            "test_b": summarize_examples(test_b),
            "unlabeled": summarize_examples(unlabeled),
            "notes": {
                "train_target": "train_merged.jsonl",
                "dev_target": "dev.jsonl",
                "ecommerce_mapping": {
                    "HPPX": "主体商品-品牌",
                    "XH": "主体商品-型号",
                    "HCCX": "主体商品-名称",
                    "MISC": "主体商品-通用属性",
                },
            },
        },
    )
    print(f"Prepared merged dataset under {args.output_dir}")


if __name__ == "__main__":
    main()
