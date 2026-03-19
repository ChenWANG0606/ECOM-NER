# Merged E-commerce NER

希望使用开源的电商表数据微调bert，用在标题实体标注任务上

当前主线代码只有三部分：

- `src/ecom_ner`：数据结构、标签映射、GlobalPointer 模型、评估逻辑
- `scripts`：数据准备、训练、推理、语料构建
- `configs`：默认训练配置


## 1. 项目目标

本项目以 GAIIC 2022 商品标题实体识别任务为主任务，额外合并 `data/中文NER数据集/ecommerce` 公开电商 NER 数据：

1. 保留 GAIIC 的细粒度语义标签作为主标签空间。
2. 将 `ecommerce` 数据通过显式映射并入训练集。

## 2. 目录结构

```text
.
├── configs/
│   └── global_pointer_baseline.json
├── scripts/
│   ├── build_corpus.py
│   ├── prepare_data.py
│   ├── predict.py
│   └── train.py
├── src/ecom_ner/
│   ├── data.py
│   ├── io.py
│   ├── labels.py
│   ├── metrics.py
│   ├── modeling.py
│   └── train_utils.py
├── data/
│   ├── 中文NER数据集/
│   └── processed/
└── outputs/
```

## 3. 数据组织与合并策略

### 3.1 原始数据

- GAIIC 主数据：`data/中文NER数据集/商品标题2022-NER/train.txt`
- GAIIC 无标注数据：`data/中文NER数据集/商品标题2022-NER/unlabeled_train_data.txt`
- GAIIC 测试标题：`preliminary_test_a` / `preliminary_test_b`
- 外部电商数据：`data/中文NER数据集/ecommerce/{train,dev,test}.txt`

### 3.2 统一中间格式

所有数据都会被转成统一 JSONL：

```json
{
  "id": "gaiic-123",
  "text": "苹果手机壳透明防摔",
  "tokens": ["苹", "果", "手", "机", "壳", "透", "明", "防", "摔"],
  "entities": [
    {"start": 0, "end": 2, "label": "主体商品-品牌", "text": "苹果"},
    {"start": 2, "end": 5, "label": "主体商品-名称", "text": "手机壳"}
  ],
  "source": "gaiic"
}
```

其中 `start/end` 为左闭右开字符区间。

### 3.3 标签体系

主标签空间来自 GAIIC 的语义标签映射，不再保留原始数字标签，例如：

- `1 -> 主体商品-品牌`
- `4 -> 主体商品-名称`
- `22 -> 配件商品-名称`
- `40 -> 其他商品-名称`

`ecommerce` 数据的标签会被映射到该语义空间。当前映射定义在 [labels.py]

- `HPPX -> 主体商品-品牌`
- `XH -> 主体商品-型号`
- `HCCX -> 主体商品-名称`
- `MISC -> 主体商品-通用属性`

这里的 `MISC` 映射是启发式的，目的是把外部数据纳入统一训练流程，而不是声称它与 GAIIC 标签完全等价。

### 3.4 训练集与验证集

`scripts/prepare_data.py` 会生成：

- `train_gaiic.jsonl`：GAIIC 训练集切分后的训练部分
- `dev.jsonl`：从 GAIIC 训练集切分出的验证集
- `train_ecommerce.jsonl`：映射后的外部电商数据
- `train_merged.jsonl`：`train_gaiic + train_ecommerce`
- `test_a.jsonl` / `test_b.jsonl`：无标注测试标题
- `unlabeled.jsonl`：GAIIC 无标注标题

默认训练目标是 `train_merged.jsonl`，默认验证目标是 `dev.jsonl`。目的是训练吸收多源信息，验证仍然贴近 GAIIC 目标分布。

## 4. 模型结构

模型采用 `Encoder + GlobalPointer`：

1. `AutoModel` 作为编码器，默认 `bert-base-chinese`，可替换为任意 HuggingFace 中文编码器。
2. 编码器输出经过 dropout。
3. 线性层投影为每个标签对应的 query / key。
4. 使用 RoPE 位置编码增强 span 建模。
5. 通过 GlobalPointer 计算每个标签下的 `(start, end)` span 分数。

对应实现位于 [modeling.py]

### 为什么仍然使用 GlobalPointer

- 商品标题天然是短文本，span 抽取比逐 token BIO 分类更直接。
- GAIIC 原任务本身就适合 span 级建模。
- 代码量小，容易看懂，也方便你以后替换编码器或损失函数。

## 5. 完整训练流程

### 5.1 安装

推荐先创建独立环境，然后在项目根目录执行：

```bash
pip install -e .
```

如果你不想安装成包，也可以直接使用：

```bash
PYTHONPATH=src python scripts/prepare_data.py
```

### 5.2 数据准备

```bash
python scripts/prepare_data.py \
  --data-root data/中文NER数据集 \
  --output-dir data/processed/merged_ner \
  --dev-ratio 0.1 \
  --seed 42
```

准备完成后，核心产物位于：

- `data/processed/merged_ner/train_merged.jsonl`
- `data/processed/merged_ner/dev.jsonl`
- `data/processed/merged_ner/labels.json`
- `data/processed/merged_ner/stats.json`

### 5.3 训练

默认配置文件在 [global_pointer_baseline.json](/Users/king/Documents/实习/虾皮/NER/configs/global_pointer_baseline.json)。

```bash
python scripts/train.py --config configs/global_pointer_baseline.json
```

训练输出目录默认是 `outputs/global_pointer_baseline/`，其中包含：

- `best_model.pt`
- `best_metrics.json`
- `train_config.json`
- `training_log.json`
- `tokenizer/`

### 5.4 推理

对准备好的 JSONL 文件预测：

```bash
python scripts/predict.py \
  --checkpoint-dir outputs/global_pointer_baseline \
  --input-file data/processed/merged_ner/test_a.jsonl \
  --output-file outputs/global_pointer_baseline/test_a_predictions.jsonl
```

对一行一个标题的纯文本预测：

```bash
python scripts/predict.py \
  --checkpoint-dir outputs/global_pointer_baseline \
  --input-file your_titles.txt \
  --output-file outputs/global_pointer_baseline/raw_predictions.jsonl
```

### 5.5 继续预训练语料

如果你后续想做领域继续预训练，可以先构建纯文本语料：

```bash
python scripts/build_corpus.py \
  --inputs \
    data/processed/merged_ner/train_merged.jsonl \
    data/processed/merged_ner/dev.jsonl \
    data/processed/merged_ner/unlabeled.jsonl \
  --output-file data/processed/merged_ner/corpus.txt
```

## 6. 训练配置说明

默认配置字段如下：

- `model_name_or_path`：编码器路径或 HuggingFace 模型名
- `max_length`：最大标题长度
- `train_batch_size` / `eval_batch_size`
- `encoder_learning_rate`：编码器学习率
- `head_learning_rate`：GlobalPointer 头部学习率
- `warmup_ratio`
- `num_epochs`
- `precision`：`fp32` / `fp16` / `bf16`
- `eval_threshold`：预测 span 时的阈值



## 7. 当前主入口

- 数据准备：[prepare_data.py](/Users/king/Documents/实习/虾皮/NER/scripts/prepare_data.py)
- 模型训练：[train.py](/Users/king/Documents/实习/虾皮/NER/scripts/train.py)
- 模型推理：[predict.py](/Users/king/Documents/实习/虾皮/NER/scripts/predict.py)
- 标签映射：[labels.py](/Users/king/Documents/实习/虾皮/NER/src/ecom_ner/labels.py)
- 模型实现：[modeling.py](/Users/king/Documents/实习/虾皮/NER/src/ecom_ner/modeling.py)

