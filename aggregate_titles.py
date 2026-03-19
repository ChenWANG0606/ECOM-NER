#!/usr/bin/env python3
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "中文NER数据集"
OUTPUT_PATH = ROOT / "chinese_ecommerce_titles.txt"


def append_titles_from_token_label_file(path: Path, sink: list[str]) -> None:
    """Files where each non-empty line is `token label`, sentences separated by blank lines."""
    tokens: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n").rstrip("\r")
            if raw == "":
                if tokens:
                    sink.append("".join(tokens))
                    tokens = []
                continue

            parts = raw.split()
            token = parts[0] if parts else raw  # raw may be whitespace the model tokenized
            tokens.append(token)

    if tokens:
        sink.append("".join(tokens))


def append_titles_from_word_per_line(path: Path, sink: list[str]) -> None:
    """Files where each token (including spaces) is on its own line, blank line splits samples."""
    tokens: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n").rstrip("\r")
            if raw == "":
                if tokens:
                    sink.append("".join(tokens))
                    tokens = []
                continue
            tokens.append(raw)

    if tokens:
        sink.append("".join(tokens))


def append_titles_from_lines(path: Path, sink: list[str]) -> None:
    """Files where each line is a complete title already."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n").rstrip("\r")
            if raw:
                sink.append(raw)


def main() -> None:
    titles_jindong: list[str] = []

    # # 商品标题2022-NER
    sku_root = DATA_ROOT / "商品标题2022-NER"
    append_titles_from_token_label_file(sku_root / "train.txt", titles_jindong)
    append_titles_from_lines(sku_root / "unlabeled_train_data.txt", titles_jindong)

    prelim_root = sku_root / "preliminary_test_a"
    append_titles_from_lines(prelim_root / "sample_per_line_preliminary_A.txt", titles_jindong)
    prelim_root = sku_root / "preliminary_test_b"
    append_titles_from_lines(prelim_root / "sample_per_line_preliminary_B.txt", titles_jindong)

    # ecommerce
    title_taobao = []
    ecommerce_root = DATA_ROOT / "ecommerce"
    for split in ("train.txt", "dev.txt", "test.txt"):
        append_titles_from_token_label_file(ecommerce_root / split, title_taobao)

    title_taobao = [title.replace(",", " ") for title in title_taobao]
    titles = title_taobao + titles_jindong
    OUTPUT_PATH.write_text("\n".join(titles) + "\n", encoding="utf-8")
    print(f"Collected {len(titles)} titles -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
