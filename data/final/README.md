# 冻结最终数据集

状态：**FROZEN**

- 问题：180
- 文档：3,500
- split：`final`
- `semantic_fact`：60
- `multi_hop_relation`：60
- `exact_file_lookup`：60

`exact_file_lookup` 进一步均分为 `title_anchor`、`date_number_lookup`、`exact_phrase_lookup`，各 20 题。

所有未被 `gold_documents` 引用的文档都按 noise / distractor 处理。Router、Evidence Judge、检索器权重和阈值不得根据该数据集的结果继续调整。

完整构建流程见 [`../../docs/DATASET_BUILD.md`](../../docs/DATASET_BUILD.md)，冻结记录见 [`FROZEN.txt`](FROZEN.txt)。
