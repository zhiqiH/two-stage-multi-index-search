# Final Dataset Package

本目录是最终数据集的纯净交付版，只包含与最终数据集构造、冻结、复现、说明直接相关的代码和文件。

不包含以下过程性内容：

- bootstrap 启动集和 smoke-test 结果
- 检索实验结果、索引、图表
- `__pycache__`、临时 tail/backup/log 文件
- 项目开发过程中的中间报告和杂乱代码

当前数据集已经冻结。冻结记录见：

```text
data/final/FROZEN.txt
```

冻结后不能再根据最终实验结果修改问题、替换样本、调 Router/Judge/retriever 参数，或改变评价口径。

## 目录结构

```text
source_code_data/
  README.md
  config.yaml
  requirements.txt

  src/
    common.py
    config.py
    data_builder.py
    text_utils.py
    final_staging_builder.py
    llm_final_data_preparer.py
    final_dataset_builder.py

  experiments/
    build_final_staging.py
    prepare_final_questions_with_llm.py
    build_final_dataset.py

  tools/
    create_dataset_report_docx.py

  data/final/
    corpus.jsonl
    questions.jsonl
    splits.json
    data_check.csv
    selection_manifest.json
    DATASET_CARD.md
    DATASET_SPEC.md
    FROZEN.txt

    manual/
      semantic_rewrite_sheet.csv
      exact_lookup_sheet.csv
      multi_hop_candidate_sheet.csv
      semantic_rewrite_sheet_llm.csv
      exact_lookup_sheet_llm.csv
      multi_hop_candidate_sheet_llm.csv

    staging/
      candidate_pool.jsonl
      corpus.jsonl
      staging_manifest.json

  docs/
    最终数据集设定与构建说明.docx
```

## 最终数据集设定

本数据集用于轻量级多索引智能体搜索项目的最终评价。它考察不同检索方法在同一个冻结语料和问题集合上的表现，包括后续的 BM25、Dense、Graph、File、Rule Router、Two-Stage Agent、Fusion 和 Oracle。

最终规模：

| 项目 | 数量 |
|---|---:|
| 问题总数 | 180 |
| 语料文档总数 | 3500 |
| split | final |
| final split 问题数 | 180 |
| 随机种子 | 42 |

任务分布：

| 任务类型 | 数量 | 目的 |
|---|---:|---|
| `semantic_fact` | 60 | 单文档语义事实检索，问题不依赖文件名或标题。 |
| `multi_hop_relation` | 60 | 双文档关系检索，问题需要连接两个 gold 文档。 |
| `exact_file_lookup` | 60 | 文件/标题/精确锚点检索，保留 File 类方法的优势。 |

`exact_file_lookup` 子类型：

| 子类型 | 数量 | 目的 |
|---|---:|---|
| `title_anchor` | 20 | 问题显式包含目标文档标题，检验标题定位能力。 |
| `date_number_lookup` | 20 | 问题包含日期或数字锚点，但不泄露 gold title。 |
| `exact_phrase_lookup` | 20 | 问题包含精确短语锚点，检验短语级定位能力。 |

语料角色：

| corpus_role | 数量 | 含义 |
|---|---:|---|
| `selected_gold` | 240 | 被最终 180 个问题的 `gold_documents` 引用。 |
| `noise` | 3260 | 未被最终 gold 引用，评测时均视为噪声/干扰文档。 |

所有非 gold 文档都作为噪声处理。评价只看 `questions.jsonl` 中的 `gold_documents`，不依赖 staging 阶段的语料角色。

## 数据来源

候选样本来自 HotpotQA：

```yaml
hotpot_dataset_name: hotpotqa/hotpot_qa
hotpot_dataset_fallback_name: hotpot_qa
hotpot_config: distractor
hotpot_split: train
```

使用 train split 的原因是本项目没有复用 HotpotQA 官方评测切分，而是从 HotpotQA 中构造一个新的、冻结的检索评价数据集。最终结果只在本项目的 final 数据集上评价。

## 构造流程

### 1. 候选池构造

入口脚本：

```powershell
python experiments/build_final_staging.py --scan-limit 30000 --seed 42
```

该步骤生成：

| 文件 | 说明 |
|---|---|
| `data/final/staging/candidate_pool.jsonl` | 270 条候选的统一 gold 结构。 |
| `data/final/staging/corpus.jsonl` | 3500 篇 staging 语料。 |
| `data/final/staging/staging_manifest.json` | staging 构造统计。 |
| `data/final/manual/semantic_rewrite_sheet.csv` | 90 条 semantic 候选。 |
| `data/final/manual/exact_lookup_sheet.csv` | 90 条 exact 候选。 |
| `data/final/manual/multi_hop_candidate_sheet.csv` | 90 条 multi-hop 候选。 |

候选规模：

| 候选类型 | 候选数 |
|---|---:|
| semantic_fact | 90 |
| multi_hop_relation | 90 |
| exact_file_lookup | 90 |
| title_anchor | 30 |
| date_number_lookup | 30 |
| exact_phrase_lookup | 30 |

`candidate_pool.jsonl` 是 gold 依据。它保存：

- `candidate_id`
- `task_type`
- `subtype`
- `question`
- `gold_documents`
- `gold_sentences`
- `source_hotpot_id`
- `metadata`

后续 LLM CSV 只决定题面文本是否合格，不重新定义 gold。

### 2. LLM 辅助构造与核验

入口脚本：

```powershell
python experiments/prepare_final_questions_with_llm.py --tasks all --all-rows
```

需要 DeepInfra API，但密钥只能通过环境变量提供：

```powershell
$env:DEEPINFRA_TOKEN="..."
```

不要把 token 写入 `config.yaml`、README、CSV 或日志。

LLM 配置：

| 用途 | 模型 | 温度 |
|---|---|---:|
| 生成/改写 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | 0.2 |
| 独立核验 | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | 0.0 |

任务策略：

- `semantic_fact`：LLM 将 gold sentence 改写为自然语义问题，然后用规则和 verifier 双重核验。
- `exact_file_lookup`：LLM 生成 final question，并按 subtype 检查标题、日期/数字、精确短语等规则。
- `multi_hop_relation`：只核验和筛选，不让 LLM 重写原问题。

LLM 全量处理结果：

| LLM 输出表 | 候选数 | accepted | needs_review | drop |
|---|---:|---:|---:|---:|
| `semantic_rewrite_sheet_llm.csv` | 90 | 76 | 14 | 0 |
| `exact_lookup_sheet_llm.csv` | 90 | 76 | 11 | 3 |
| `multi_hop_candidate_sheet_llm.csv` | 90 | 63 | 22 | 5 |

只有 `status=accepted` 的候选可以进入最终数据集。`needs_review` 和 `drop` 均不能进入 final。

### 3. 最终抽样与冻结

入口脚本：

```powershell
python experiments/build_final_dataset.py --freeze
```

该脚本执行：

1. 读取三份 `*_llm.csv`。
2. 只保留 `status=accepted` 的候选。
3. 使用固定 seed `42` 分层随机抽样。
4. 从候选池 `candidate_pool.jsonl` 读取 gold 信息。
5. 从 staging corpus 复制 3500 篇文档。
6. 将最终 gold 文档标为 `selected_gold`，其余标为 `noise`。
7. 写出 final 数据文件。
8. 写出 `selection_manifest.json`。
9. 创建 `FROZEN.txt`。

最终抽样池：

| 抽样层 | accepted 候选数 | 最终抽取数 |
|---|---:|---:|
| semantic_fact | 76 | 60 |
| multi_hop_relation | 63 | 60 |
| title_anchor | 29 | 20 |
| date_number_lookup | 26 | 20 |
| exact_phrase_lookup | 21 | 20 |

最终数据文件：

| 文件 | 说明 |
|---|---|
| `data/final/questions.jsonl` | 最终 180 个问题。 |
| `data/final/corpus.jsonl` | 最终 3500 篇语料。 |
| `data/final/splits.json` | 仅包含 `final`，共 180 个 question_id。 |
| `data/final/data_check.csv` | 最终质量检查表。 |
| `data/final/selection_manifest.json` | 输入 hash、输出 hash、抽样结果、candidate 映射。 |
| `data/final/DATASET_CARD.md` | 数据集卡片。 |
| `data/final/FROZEN.txt` | 冻结记录。 |

## 数据结构

### questions.jsonl

每行是一个问题对象：

```json
{
  "question_id": "sf_0001",
  "question": "...",
  "task_type": "semantic_fact",
  "gold_documents": ["doc_..."],
  "gold_sentences": [
    {"doc_id": "doc_...", "sent_id": 0, "text": "..."}
  ],
  "source_hotpot_id": "...",
  "split": "final",
  "quality_checked": true,
  "metadata": {
    "candidate_id": "...",
    "construction": "...",
    "subtype": "...",
    "source_sheet": "...",
    "source_status": "accepted"
  }
}
```

重要约束：

- `split` 必须是 `final`。
- `quality_checked` 必须是 `true`。
- `gold_documents` 必须存在于 `corpus.jsonl`。
- `gold_sentences` 必须非空。
- `metadata.candidate_id` 关联回 `candidate_pool.jsonl` 和 `selection_manifest.json`。

### corpus.jsonl

每行是一个文档对象：

```json
{
  "doc_id": "doc_...",
  "title": "...",
  "sentences": ["..."],
  "full_text": "...",
  "source_question_ids": ["sf_0001"],
  "metadata": {
    "corpus_role": "selected_gold",
    "final_gold_question_ids": ["sf_0001"],
    "staging_corpus_role": "candidate_gold"
  }
}
```

`metadata.corpus_role` 只有两类：

- `selected_gold`
- `noise`

## 最终校验结果

最终校验通过：

```text
questions: 180
semantic_fact: 60
multi_hop_relation: 60
exact_file_lookup: 60

title_anchor: 20
date_number_lookup: 20
exact_phrase_lookup: 20

corpus: 3500
selected_gold: 240
noise: 3260
splits.final: 180
manual_flags: 0
```

`manual_flags=0` 表示 final questions 中不再残留 `requires_manual_rewrite=true` 或 `requires_manual_review=true`。

## 冻结记录

冻结文件：

```text
data/final/FROZEN.txt
```

冻结内容摘要：

```text
FINAL DATASET FROZEN
timestamp_utc: 2026-07-15T13:08:00.408059+00:00
seed: 42
manifest_path: data\final\selection_manifest.json
manifest_sha256: 010b46958fa3b0d22c09876dfe117b9bcad903f24f3857d0b439428674580634
question_count: 180
corpus_documents: 3500
method_tuning_on_final_allowed: false
```

冻结后规则：

- 不得根据 final 实验失败样本修改问题。
- 不得根据 final 指标调 Router/Judge/retriever 参数。
- 不得替换 final 样本。
- 如发现客观数据错误，必须记录 changelog，且不得以提升某个方法指标为目标。

## Hash 与可追溯性

输入文件 hash：

| 输入 | SHA-256 |
|---|---|
| candidate_pool | `be0fdf94aa3ccab2367e6296ac23913ae07e296de199b907124abb576727dcd9` |
| staging_corpus | `b3c32c0c01077195bea3d5d452bd354b77727d1db1895263d469ca7c047f3237` |
| semantic_sheet | `79239e51be007b880f367c9f5345cb24250d8bff3e8673c23d9a23b9d58b9d7b` |
| exact_sheet | `4e133edd150940739d17972c5ab4fea2d02f9826797782cdf637ce49eec69c57` |
| multi_hop_sheet | `7d9703cf54d0889e1c03eef14b23e6193df6917dc72ba23233c2c908ecd294eb` |

输出文件 hash：

| 输出 | SHA-256 |
|---|---|
| corpus | `f159298301a23546ab284ecba0078aab714d618876a54650203c46a543847d6e` |
| questions | `b3c26d798ddaa6416bcbbefe1e2b6bd0a2c8564b7db10fb8097c3dac28ff63c4` |
| splits | `126a3f86812d03599636d3aad4fcb557171d4a4e18c2bb36321989c588e998ef` |
| data_check | `57ea215326da64de59af00beb571625ac8e4e13433b4839ee8af2052fe753241` |
| dataset_card | `46685915015c93e9d89a4d913937f9f19efd55f38e62096d23521da136c7d5ea` |

## 开发说明书

### 代码模块职责

| 文件 | 职责 |
|---|---|
| `src/common.py` | JSONL 读写、统一数据结构辅助函数。 |
| `src/config.py` | YAML 配置读取。 |
| `src/data_builder.py` | HotpotQA context/supporting facts 解析、稳定 doc_id 生成。 |
| `src/text_utils.py` | 文本归一化、数字/日期/实体/关键词工具。 |
| `src/final_staging_builder.py` | 构造 final staging 候选池和 3500 篇 staging corpus。 |
| `src/llm_final_data_preparer.py` | 调用 DeepInfra LLM 进行题面生成、核验、返修、筛选。 |
| `src/final_dataset_builder.py` | 从 accepted 候选中随机抽样，生成并冻结 final 数据集。 |
| `experiments/build_final_staging.py` | staging 构造入口。 |
| `experiments/prepare_final_questions_with_llm.py` | LLM 准备入口。 |
| `experiments/build_final_dataset.py` | final 构建与冻结入口。 |
| `tools/create_dataset_report_docx.py` | 生成 Word 版数据集说明文档。 |

### 环境安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

最小依赖：

- `datasets`
- `openai`
- `pyyaml`

### 复现 final 数据集

如果只想从已存在的 LLM accepted CSV 复现最终冻结数据集：

```powershell
python experiments/build_final_dataset.py --freeze --overwrite
```

该命令不会调用 DeepInfra，也不需要 API token。

### 重新生成 staging 候选

通常不应在冻结后执行。如果需要完全重建候选池：

```powershell
python experiments/build_final_staging.py --scan-limit 30000 --seed 42
```

注意：这会改变 staging 文件，属于重新构造数据集，不应在 final 实验后为了提升指标而执行。

### 重新运行 LLM 处理

通常不应在冻结后执行。如果确实需要重新处理候选：

```powershell
$env:DEEPINFRA_TOKEN="..."
python experiments/prepare_final_questions_with_llm.py --tasks all --all-rows
Remove-Item Env:\DEEPINFRA_TOKEN
```

安全要求：

- token 只放环境变量。
- 不写入 `config.yaml`。
- 不写入 README。
- 不写入 CSV 或 JSONL。
- 运行后清除环境变量。

### 生成说明文档

```powershell
python tools/create_dataset_report_docx.py
```

当前已生成的 Word 文档位于：

```text
docs/最终数据集设定与构建说明.docx
```

## 后续实验边界

可以使用本数据集运行正式检索实验：

- BM25
- Dense
- Graph
- File
- Rule Router
- Two-Stage Agent
- Fusion
- Oracle

统一评价建议使用：

```text
Complete@1 / Complete@3 / Complete@5
```

但必须遵守：

- final 数据集不是验证集。
- final 数据集不能用于调参。
- final 错误分析只能用于论文/报告中的分析，不能反向修改数据或方法参数。
- 如果需要继续改 Router/Judge/参数，应回到 bootstrap 或独立验证集，不能用 final 结果。
