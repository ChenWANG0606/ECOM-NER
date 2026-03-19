import random
import os
import json
from typing import List

import jieba
from tqdm import tqdm, trange
LABEL2MEANING_MAP = {
    # 主体商品，即上架的商品
    "1": "主体商品-品牌",
    "2": "主体商品-系列",
    "3": "主体商品-型号",
    "4": "主体商品-名称",
    "5": "主体商品-用途",
    "6": "主体商品-时间",
    "7": "主体商品-地点",
    "8": "主体商品-人群",
    "9": "主体商品-用途名词",
    "10": "主体商品-周边",
    "11": "主体商品-功能",
    "12": "主体商品-材料",
    "13": "主体商品-样式",
    "14": "主体商品-风格",
    "15": "主体商品-产地",
    "16": "主体商品-颜色",
    "17": "主体商品-味道",
    "18": "主体商品-尺寸",
    # 配件商品，即主体商品包含的子商品
    "19": "配件商品-品牌",
    "20": "配件商品-系列",
    "21": "配件商品-型号",
    "22": "配件商品-名称",
    "23": "配件商品-用途",
    "24": "配件商品-时间",
    "25": "配件商品-地点",
    "26": "配件商品-人群",
    "27": "配件商品-用途名词",  # 数据中无该类别
    "28": "配件商品-周边",
    "29": "配件商品-功能",
    "30": "配件商品-材料",
    "31": "配件商品-样式",
    "32": "配件商品-风格",
    "33": "配件商品-产地",
    "34": "配件商品-颜色",
    "35": "配件商品-味道",
    "36": "配件商品-尺寸",
    # 其他商品，包括适用商品、赠送商品
    "37": "其他商品-品牌",
    "38": "其他商品-系列",
    "39": "其他商品-型号",
    "40": "其他商品-名称",
    "41": "其他商品-用途",
    "42": "其他商品-时间",
    "43": "其他商品-地点",
    "44": "其他商品-人群",
    "45": "其他商品-用途名词",  # 数据中无该类别
    "46": "其他商品-周边",
    "47": "其他商品-功能",
    "48": "其他商品-材料",
    "49": "其他商品-样式",
    "50": "其他商品-风格",
    "51": "其他商品-产地",
    "52": "其他商品-颜色",
    "53": "其他商品-味道",
    "54": "其他商品-尺寸",
}

MEANING2LABEL_MAP = {v: k for k, v in LABEL2MEANING_MAP.items()}

def read_titles(data_dir):
    """Read e-commerce titles from a text file, one title per line."""
    file_name = data_dir
    with open(file_name, 'r') as f:
        titles = [line.rstrip('\n') for line in f]

    random.shuffle(titles)
    return titles

def generate_examples_from_lines(input_file):
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            yield (i, dict(tokens=list(line)))

def generate_examples(data_path):
    sentence_counter = 0
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    
    current_words = []
    current_labels = []
    for row in lines:
        row = row.rstrip("\n")
        if row != "":
            token, label = row[0], row[2:]
            current_words.append(token)
            current_labels.append(label)
        else:
            if not current_words:
                continue
            assert len(current_words) == len(current_labels), "word len doesn't match label length"
            sentence = (
                sentence_counter,
                {
                    "id": str(sentence_counter),
                    "tokens": current_words,
                    "ner_tags": current_labels,
                },
            )
            sentence_counter += 1
            current_words = []
            current_labels = []
            yield sentence

    # if something remains:
    if current_words:
        sentence = (
            sentence_counter,
            {
                "id": str(sentence_counter),
                "tokens": current_words,
                "ner_tags": current_labels,
            },
        )
        yield sentence


def get_spans_bio(tags, id2label=None):
    """Gets entities from sequence.
    Args:
        tags (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> tags = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_spans_bio(tags)
        # output [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(tags):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx     # FIXED: 该函数无法提取由"B-X"标记的单个token实体
            chunk[0] = tag.split('-')[1]
            if indx == len(tags) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(tags) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks




def is_chinese(word: str):
    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False
    # word like '180' or '身高' or '神'
    for char in word:
        char = ord(char)
        if not _is_chinese_char(char):
            return 0
    return 1


def get_chinese_word(tokens: List[str]):
    word_set = set()

    for token in tokens:
        chinese_word = len(token) > 1 and is_chinese(token)
        if chinese_word:
            word_set.add(token)
    word_list = list(word_set)
    return word_list


def add_sub_symbol(bert_tokens: List[str], chinese_word_set: set()):
    if not chinese_word_set:
        return bert_tokens
    max_word_len = max([len(w) for w in chinese_word_set])

    bert_word = bert_tokens
    start, end = 0, len(bert_word)
    while start < end:
        single_word = True
        if is_chinese(bert_word[start]):
            l = min(end - start, max_word_len)
            for i in range(l, 1, -1):
                whole_word = "".join(bert_word[start : start + i])
                if whole_word in chinese_word_set:
                    for j in range(start + 1, start + i):
                        bert_word[j] = "##" + bert_word[j]
                    start = start + i
                    single_word = False
                    break
        if single_word:
            start += 1
    return bert_word


def prepare_ref(lines: List[str], ltp_tokenizer, bert_tokenizer: BertTokenizerZh):
    seg_res = []
    print(f"Using {'ltp' if ltp_tokenizer is not None else 'jieba'}")
    for i in trange(0, len(lines), 100):
        if ltp_tokenizer is not None:
            # ltp
            res = ltp_tokenizer.seg(lines[i : i + 100])[0]
        else:
            # jieba
            res = []
            for line in lines[i : i + 100]:
                seg = jieba.lcut(line)
                res.append(seg)
        res = [get_chinese_word(r) for r in res]
        seg_res.extend(res)
    assert len(seg_res) == len(lines)

    bert_res = []
    for i in trange(0, len(lines), 100):
        res = bert_tokenizer(lines[i : i + 100], add_special_tokens=True, truncation=True, max_length=512)
        bert_res.extend(res["input_ids"])
    assert len(bert_res) == len(lines)

    ref_ids = []
    for input_ids, chinese_word in tqdm(zip(bert_res, seg_res), total=len(bert_res)):
        input_tokens = []
        for id in input_ids:
            token = bert_tokenizer._convert_id_to_token(id)
            input_tokens.append(token)
        input_tokens = add_sub_symbol(input_tokens, chinese_word)
        ref_id = []
        # We only save pos of chinese subwords start with ##, which mean is part of a whole word.
        for i, token in enumerate(input_tokens):
            if token[:2] == "##":
                clean_token = token[2:]
                # save chinese tokens' pos
                if len(clean_token) == 1 and _is_chinese_char(ord(clean_token)):
                    ref_id.append(i)
        ref_ids.append(ref_id)

    assert len(ref_ids) == len(bert_res)

    return seg_res, ref_ids
