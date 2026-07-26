from __future__ import annotations

import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "reports" / "最终数据集设定与构建说明.docx"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def xml_text(value: object) -> str:
    return escape(str(value))


def run(text: object, *, bold: bool = False) -> str:
    props = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f"<w:r>{props}<w:t xml:space=\"preserve\">{xml_text(text)}</w:t></w:r>"


def para(text: object = "", *, style: str | None = None, bold: bool = False) -> str:
    ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
    return f"<w:p>{ppr}{run(text, bold=bold)}</w:p>"


def bullet(text: object) -> str:
    return para(f"• {text}")


def table(rows: list[list[object]]) -> str:
    cells = []
    for row_index, row in enumerate(rows):
        tr = []
        for cell in row:
            shading = "<w:shd w:fill=\"D9EAF7\"/>" if row_index == 0 else ""
            tr.append(
                "<w:tc>"
                f"<w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/>{shading}</w:tcPr>"
                f"{para(cell, bold=(row_index == 0))}"
                "</w:tc>"
            )
        cells.append(f"<w:tr>{''.join(tr)}</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"808080\"/>"
        "<w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"808080\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"808080\"/>"
        "<w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"808080\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"808080\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"808080\"/>"
        "</w:tblBorders></w:tblPr>"
        f"{''.join(cells)}"
        "</w:tbl>"
    )


def code_block(lines: list[str]) -> str:
    return "".join(para(line, style="Code") for line in lines)


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="21"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:rPr><w:b/><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Code">
    <w:name w:val="Code"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Consolas" w:eastAsia="Microsoft YaHei"/><w:sz w:val="18"/></w:rPr>
  </w:style>
  <w:style w:type="table" w:default="1" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:tblPr><w:tblBorders>
      <w:top w:val="single" w:sz="4" w:space="0" w:color="808080"/>
      <w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>
      <w:bottom w:val="single" w:sz="4" w:space="0" w:color="808080"/>
      <w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>
      <w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/>
      <w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/>
    </w:tblBorders></w:tblPr>
  </w:style>
</w:styles>"""


def document_xml(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""


def write_docx(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        )
        docx.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        )
        docx.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""",
        )
        docx.writestr("word/document.xml", document_xml(body))
        docx.writestr("word/styles.xml", styles_xml())
        now = datetime.now(timezone.utc).isoformat()
        docx.writestr(
            "docProps/core.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>最终数据集设定与构建说明</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""",
        )
        docx.writestr(
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>Codex</Application>
</Properties>""",
        )


def main() -> int:
    questions = read_jsonl(PROJECT_ROOT / "data/final/questions.jsonl")
    corpus = read_jsonl(PROJECT_ROOT / "data/final/corpus.jsonl")
    manifest = json.loads(read_text(PROJECT_ROOT / "data/final/selection_manifest.json"))
    frozen_text = read_text(PROJECT_ROOT / "data/final/FROZEN.txt")

    task_counts = Counter(q["task_type"] for q in questions)
    exact_counts = Counter(q["metadata"].get("subtype") for q in questions if q["task_type"] == "exact_file_lookup")
    corpus_roles = Counter(doc["metadata"]["corpus_role"] for doc in corpus)
    source_status_counts = {
        "semantic": "accepted=76; needs_review=14",
        "exact": "accepted=76; drop=3; needs_review=11",
        "multi_hop": "accepted=63; drop=5; needs_review=22",
    }

    body_parts: list[str] = []
    body_parts.append(para("最终数据集设定与构建说明", style="Title"))
    body_parts.append(para("轻量级多索引智能体搜索项目", bold=True))
    body_parts.append(para(f"生成时间（UTC）: {datetime.now(timezone.utc).isoformat()}"))
    body_parts.append(para(f"项目路径: {PROJECT_ROOT}"))
    body_parts.append(para(f"文档输出路径: {OUTPUT_PATH}"))

    body_parts.append(para("1. 数据集定位", style="Heading1"))
    body_parts.append(para("本数据集是 Two-Stage Lightweight Task-Aware Multi-Index Agentic Search 项目的最终评价数据集。它用于比较 BM25、Dense、Graph、File、Rule Router、Two-Stage Agent、Fusion 和 Oracle 等检索方法在同一冻结语料与问题集合上的表现。"))
    body_parts.append(para("数据集已经冻结，冻结记录位于 data/final/FROZEN.txt。冻结后不得再基于最终实验结果修改题目、替换样本、调整 Router/Judge/retriever 参数或改变评价口径。"))
    body_parts.append(bullet("Bootstrap 启动集只用于早期 smoke test，不介入 final 实验结果。"))
    body_parts.append(bullet("Final 数据集没有 dev/test split，所有问题的 split 均为 final。"))
    body_parts.append(bullet("Final 数据集只作为评价集使用，不再作为调参集或验证集使用。"))

    body_parts.append(para("2. 最终规模与组成", style="Heading1"))
    body_parts.append(table([
        ["项目", "最终设定"],
        ["问题总数", len(questions)],
        ["语料文档总数", len(corpus)],
        ["split", "final: 180"],
        ["随机抽样 seed", manifest["seed"]],
        ["冻结时间 UTC", "2026-07-15T13:08:00.408059+00:00"],
    ]))
    body_parts.append(table([
        ["任务类型", "数量", "说明"],
        ["semantic_fact", task_counts["semantic_fact"], "单文档语义事实问题，由 LLM 将事实句改写为自然语义问题并经核验。"],
        ["multi_hop_relation", task_counts["multi_hop_relation"], "双文档关系问题，保留 HotpotQA 原问题，只做自动核验和筛选，不让 LLM 重写。"],
        ["exact_file_lookup", task_counts["exact_file_lookup"], "强调文件/标题/精确短语/数字日期查找优势的问题。"],
    ]))
    body_parts.append(table([
        ["exact_file_lookup 子类型", "数量", "设计目的"],
        ["title_anchor", exact_counts["title_anchor"], "问题显式给出目标文档标题，保留 File 工具通过标题定位的优势。"],
        ["date_number_lookup", exact_counts["date_number_lookup"], "问题含日期或数字锚点，但不泄露 gold title，检验精确值查找。"],
        ["exact_phrase_lookup", exact_counts["exact_phrase_lookup"], "问题含精确短语锚点，要求系统通过短语定位相关文档。"],
    ]))

    body_parts.append(para("3. 数据来源与语料设定", style="Heading1"))
    body_parts.append(para("候选样本来自 HotpotQA，配置为 hotpotqa/hotpot_qa / distractor / train。使用 train split 的原因是本项目不直接复用 HotpotQA 官方评测切分，而是构造一个独立冻结的检索评价集。"))
    body_parts.append(para("语料库最终固定为 3500 篇文档。构造时先保证所有最终 gold_documents 都在语料中，再将其余文档作为噪声/干扰文档。最终语料角色如下："))
    body_parts.append(table([
        ["corpus_role", "数量", "含义"],
        ["selected_gold", corpus_roles["selected_gold"], "被最终 180 个问题的 gold_documents 引用的文档。"],
        ["noise", corpus_roles["noise"], "未被最终 gold_documents 引用的文档，评测中均视为噪声/干扰。"],
    ]))
    body_parts.append(para("最终有 240 篇 selected_gold 文档和 3260 篇 noise 文档。这个设定使检索系统必须在较大噪声语料中找回少量目标文档，而不是只在候选支持文档内排序。"))

    body_parts.append(para("4. 候选池构造", style="Heading1"))
    body_parts.append(para("候选池由 experiments/build_final_staging.py 构建。该阶段只生成 staging 候选，不生成 final questions.jsonl，不运行正式实验。"))
    body_parts.append(table([
        ["候选文件", "内容"],
        ["data/final/staging/candidate_pool.jsonl", "270 条候选问题的统一 gold 结构。"],
        ["data/final/staging/corpus.jsonl", "3500 篇 staging 语料。"],
        ["data/final/manual/semantic_rewrite_sheet.csv", "90 条 semantic_fact 候选。"],
        ["data/final/manual/multi_hop_candidate_sheet.csv", "90 条 multi_hop_relation 候选。"],
        ["data/final/manual/exact_lookup_sheet.csv", "90 条 exact_file_lookup 候选，三类各 30。"],
    ]))
    body_parts.append(para("candidate_pool.jsonl 是 gold 依据，保存 candidate_id、task_type、subtype、gold_documents、gold_sentences、source_hotpot_id 和 metadata。后续 LLM CSV 只决定题面是否合格以及最终题面文本，不重新定义 gold。"))

    body_parts.append(para("5. LLM 辅助构造与自动核验", style="Heading1"))
    body_parts.append(para("LLM 使用 DeepInfra OpenAI-compatible API。API 密钥只通过环境变量 DEEPINFRA_TOKEN 使用，没有写入项目文件。"))
    body_parts.append(table([
        ["用途", "模型", "温度", "说明"],
        ["问题生成/改写", "Qwen/Qwen3-235B-A22B-Instruct-2507", "0.2", "用于 semantic_fact 改写和 exact_file_lookup 题面生成。"],
        ["独立核验", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "0.0", "用于验证题目是否可答、是否符合 subtype、是否引入新事实。"],
    ]))
    body_parts.append(para("任务级策略如下："))
    body_parts.append(bullet("semantic_fact：LLM 将 gold sentence 改写成自然语义问题；程序规则检查是否过度复制原句、是否含 document/file/evidence 等文件查找词；再由 verifier 判断是否可由 gold sentence 回答、是否自然、是否引入新事实。"))
    body_parts.append(bullet("exact_file_lookup：LLM 生成 final_question；不同 subtype 有不同约束。title_anchor 必须显式包含 gold title 并体现 document/file 查找；date_number_lookup 必须包含日期/数字锚点且不能泄露 gold title；exact_phrase_lookup 必须包含引号中的精确短语且不能泄露 gold title。"))
    body_parts.append(bullet("multi_hop_relation：不让 LLM 重写问题，只验证是否确实需要两个 gold 文档、两个支撑句是否相关、答案是否没有泄露、给定证据是否足够。"))
    body_parts.append(table([
        ["LLM 输出表", "候选数", "accepted", "needs_review", "drop"],
        ["semantic_rewrite_sheet_llm.csv", 90, 76, 14, 0],
        ["exact_lookup_sheet_llm.csv", 90, 76, 11, 3],
        ["multi_hop_candidate_sheet_llm.csv", 90, 63, 22, 5],
    ]))
    body_parts.append(para("只有 status=accepted 的行可以进入最终随机抽样。needs_review 和 drop 均不进入 final 数据集。"))

    body_parts.append(para("6. 最终随机抽样与冻结", style="Heading1"))
    body_parts.append(para("最终数据集由 experiments/build_final_dataset.py 构建。脚本读取 candidate_pool.jsonl 作为 gold 依据，读取三份 LLM accepted CSV 作为题面来源，并使用固定 seed=42 做分层随机抽样。"))
    body_parts.append(table([
        ["抽样层", "accepted 候选数", "最终抽取数"],
        ["semantic_fact", 76, 60],
        ["multi_hop_relation", 63, 60],
        ["title_anchor", 29, 20],
        ["date_number_lookup", 26, 20],
        ["exact_phrase_lookup", 21, 20],
    ]))
    body_parts.append(para("构建命令如下："))
    body_parts.append(code_block(["python experiments/build_final_dataset.py --freeze"]))
    body_parts.append(para("脚本写入 selection_manifest.json，记录输入文件 hash、输出文件 hash、seed、最终 selected candidate_id 与 final_question_id 的映射。FROZEN.txt 创建后，NOT_FROZEN_YET.txt 被移除，数据集进入不可变状态。"))

    body_parts.append(para("7. 统一数据结构", style="Heading1"))
    body_parts.append(para("questions.jsonl 中每行是一个问题对象，核心字段如下："))
    body_parts.append(table([
        ["字段", "说明"],
        ["question_id", "最终问题 ID，例如 sf_0001、mh_0001、ex_title_0001。"],
        ["question", "最终题面文本。"],
        ["task_type", "semantic_fact / multi_hop_relation / exact_file_lookup。"],
        ["gold_documents", "正确文档 doc_id 列表。"],
        ["gold_sentences", "支撑句，含 doc_id、sent_id、text。"],
        ["source_hotpot_id", "来源 HotpotQA 样本 ID。"],
        ["split", "固定为 final。"],
        ["quality_checked", "固定为 true。"],
        ["metadata", "包含 candidate_id、construction、subtype、LLM 模型、source_sheet、source_status 等。"],
    ]))
    body_parts.append(para("corpus.jsonl 中每行是一个文档对象，核心字段为 doc_id、title、sentences、full_text、source_question_ids、metadata。final 版本中 metadata.corpus_role 只保留 selected_gold 或 noise 两类。"))

    body_parts.append(para("8. 校验门槛与最终校验结果", style="Heading1"))
    body_parts.append(para("build_final_dataset.py 在写入前执行硬性校验：问题总数必须为 180；corpus 必须为 3500；三类任务必须各 60；exact 三个 subtype 必须各 20；所有 gold_documents 必须存在于 corpus；所有问题必须 split=final 且 quality_checked=true；每题必须有非空 question 与 gold_sentences。"))
    body_parts.append(table([
        ["校验项", "结果"],
        ["questions", "180"],
        ["semantic_fact", "60"],
        ["multi_hop_relation", "60"],
        ["exact_file_lookup", "60"],
        ["title_anchor/date_number_lookup/exact_phrase_lookup", "20 / 20 / 20"],
        ["corpus documents", "3500"],
        ["splits.final", "180"],
        ["manual flags", "0"],
    ]))

    body_parts.append(para("9. 文件清单与哈希", style="Heading1"))
    body_parts.append(table([
        ["文件", "作用"],
        ["data/final/questions.jsonl", "最终 180 个问题。"],
        ["data/final/corpus.jsonl", "最终 3500 篇语料。"],
        ["data/final/splits.json", "final split 列表。"],
        ["data/final/data_check.csv", "最终质量检查表。"],
        ["data/final/selection_manifest.json", "抽样清单、输入/输出 hash 与 candidate 映射。"],
        ["data/final/DATASET_CARD.md", "数据集卡片。"],
        ["data/final/FROZEN.txt", "冻结记录。"],
    ]))
    body_parts.append(para("输入 hash："))
    body_parts.append(table([["输入", "SHA-256"], *[[k, v] for k, v in manifest["input_hashes"].items()]]))
    body_parts.append(para("输出 hash："))
    body_parts.append(table([["输出", "SHA-256"], *[[k, v] for k, v in manifest["output_hashes"].items()]]))
    body_parts.append(para("冻结记录摘录："))
    body_parts.append(code_block(frozen_text.strip().splitlines()))

    body_parts.append(para("10. 后续实验使用边界", style="Heading1"))
    body_parts.append(bullet("可以在 final 数据集上运行 BM25、Dense、Graph、File、Rule Router、Two-Stage Agent、Fusion、Oracle，并统一评价 @1/@3/@5。"))
    body_parts.append(bullet("不能根据 final 失败样本修改题面、替换样本、调 Router/Judge/retriever 参数。"))
    body_parts.append(bullet("如发现客观数据错误，只能以 changelog 形式记录修正，且不得以提升某个方法指标为目标。"))
    body_parts.append(bullet("所有正式结果均应引用 selection_manifest.json 与 FROZEN.txt，确保实验可追溯。"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_docx(OUTPUT_PATH, "\n".join(body_parts))
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
