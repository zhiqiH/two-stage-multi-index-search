# 数据目录

本项目只有一个正式数据集：`data/final/`。

主要文件：

- `final/corpus.jsonl`：3,500 篇可检索文档。
- `final/questions.jsonl`：180 个冻结问题及离线评价标签。
- `final/splits.json`：全部问题均属于 `final` split。
- `final/data_check.csv`：逐题质量检查。
- `final/manual/`：LLM 处理前后的候选审核表。
- `final/staging/`：候选池和构建阶段语料。

字段、来源、构建命令和 SHA-256 记录见：

- [`../docs/DATASET_BUILD.md`](../docs/DATASET_BUILD.md)
- [`final/DATASET_CARD.md`](final/DATASET_CARD.md)
- [`final/DATASET_SPEC.md`](final/DATASET_SPEC.md)

`gold_documents` 和 `gold_sentences` 只能用于离线评价。运行时检索通过 `src/common.py` 转换为公开视图，不向 Router、Judge 或检索器暴露 gold 字段。
