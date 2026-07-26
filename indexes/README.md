# 索引目录

`final/` 保存冻结语料对应的可复用索引：

- `dense_embeddings.npy`
- `dense_doc_ids.json`
- `file_fts.sqlite`
- `graph_path.pkl`

这些文件是构建阶段缓存，不是运行时指标。索引构建耗时应与在线查询时延分开报告。
