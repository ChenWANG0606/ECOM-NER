"""标签定义与跨数据集标签归一化规则。

这个文件定义了新项目使用的统一语义标签空间，并说明多个原始数据集如何
被映射到同一个任务上。

当前提供的数据和函数：
- ``GAIIC_LABEL_MAP``：把 GAIIC 原始数字标签转换成可读的中文语义标签。
- ``ECOMMERCE_LABEL_MAP``：把公开 ``ecommerce`` 数据集的标签映射到
  GAIIC 风格的语义空间。
- ``UNIFIED_LABELS``：训练和推理最终使用的统一标签列表。
- ``normalize_bio_tag``：把指定来源数据集的原始 BIO 标签转换成统一 BIO
  形式。
- ``label_to_id_map``：构建模型训练和解码所需的标签到索引映射。

这个文件没有 ``main`` 函数，因为它是纯定义模块，主要被数据转换、数据集
构建、训练和预测逻辑导入使用。
"""

from __future__ import annotations

from typing import Dict, List


GAIIC_LABEL_MAP: Dict[str, str] = {
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
    "19": "配件商品-品牌",
    "20": "配件商品-系列",
    "21": "配件商品-型号",
    "22": "配件商品-名称",
    "23": "配件商品-用途",
    "24": "配件商品-时间",
    "25": "配件商品-地点",
    "26": "配件商品-人群",
    "28": "配件商品-周边",
    "29": "配件商品-功能",
    "30": "配件商品-材料",
    "31": "配件商品-样式",
    "32": "配件商品-风格",
    "33": "配件商品-产地",
    "34": "配件商品-颜色",
    "35": "配件商品-味道",
    "36": "配件商品-尺寸",
    "37": "其他商品-品牌",
    "38": "其他商品-系列",
    "39": "其他商品-型号",
    "40": "其他商品-名称",
    "41": "其他商品-用途",
    "42": "其他商品-时间",
    "43": "其他商品-地点",
    "44": "其他商品-人群",
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

# ``ecommerce`` 数据集的标签采用启发式方式映射到 GAIIC 语义空间。
ECOMMERCE_LABEL_MAP: Dict[str, str] = {
    "HPPX": "主体商品-品牌",
    "XH": "主体商品-型号",
    "HCCX": "主体商品-名称",
    "MISC": "主体商品-通用属性",
}

UNIFIED_LABELS: List[str] = list(GAIIC_LABEL_MAP.values()) + ["主体商品-通用属性"]


def normalize_bio_tag(tag: str, source: str) -> str:
    if tag == "O":
        return tag
    prefix, raw_label = tag.split("-", 1)
    if source == "gaiic":
        label = GAIIC_LABEL_MAP[raw_label]
    elif source == "ecommerce":
        label = ECOMMERCE_LABEL_MAP[raw_label]
    else:
        raise ValueError(f"Unsupported source: {source}")
    return f"{prefix}-{label}"


def label_to_id_map(labels: List[str] | None = None) -> Dict[str, int]:
    names = labels or UNIFIED_LABELS
    return {label: idx for idx, label in enumerate(names)}
