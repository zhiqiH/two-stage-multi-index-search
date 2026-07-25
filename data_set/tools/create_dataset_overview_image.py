from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data" / "final"
PNG_OUT = PROJECT_ROOT / "最终数据集结构图.png"
SVG_OUT = PROJECT_ROOT / "最终数据集结构图.svg"


def load_stats() -> tuple[Counter[str], Counter[str], Counter[str], dict]:
    questions = [
        json.loads(line)
        for line in (DATA_ROOT / "questions.jsonl").open(encoding="utf-8")
    ]
    corpus = [
        json.loads(line)
        for line in (DATA_ROOT / "corpus.jsonl").open(encoding="utf-8")
    ]
    manifest = json.loads(
        (DATA_ROOT / "selection_manifest.json").read_text(encoding="utf-8")
    )
    task_counts = Counter(q["task_type"] for q in questions)
    exact_counts = Counter(
        q["metadata"].get("subtype")
        for q in questions
        if q["task_type"] == "exact_file_lookup"
    )
    role_counts = Counter(d["metadata"].get("corpus_role") for d in corpus)
    return task_counts, exact_counts, role_counts, manifest


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\Dengb.ttf") if bold else Path(r"C:\Windows\Fonts\Deng.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def build_png(
    task_counts: Counter[str],
    exact_counts: Counter[str],
    role_counts: Counter[str],
    manifest: dict,
) -> None:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), "#F7F9FC")
    draw = ImageDraw.Draw(image)

    navy = "#17233B"
    blue = "#2F6FED"
    teal = "#1F9A8A"
    green = "#2F9E44"
    orange = "#F08C00"
    red = "#D9480F"
    slate = "#40516B"
    muted = "#68758B"
    line = "#C9D4E5"
    white = "#FFFFFF"
    soft_blue = "#EAF1FF"
    soft_teal = "#E7F7F4"
    soft_green = "#EAF8EE"
    soft_orange = "#FFF4E0"
    soft_red = "#FFF0EA"

    f_title = font(54, bold=True)
    f_subtitle = font(26)
    f_section = font(30, bold=True)
    f_card_title = font(25, bold=True)
    f_body = font(21)
    f_small = font(18)
    f_tiny = font(15)
    f_num_small = font(34, bold=True)

    def rounded_rect(xy, radius=18, fill=white, outline="#D9E2EF", stroke_width=2):
        draw.rounded_rectangle(
            xy, radius=radius, fill=fill, outline=outline, width=stroke_width
        )

    def text(x, y, value, face, fill=navy, anchor=None, align="left"):
        draw.text((x, y), value, font=face, fill=fill, anchor=anchor, align=align)

    def center_text(box, value, face, fill=navy):
        x1, y1, x2, y2 = box
        bbox = draw.multiline_textbbox(
            (0, 0), value, font=face, spacing=6, align="center"
        )
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.multiline_text(
            (x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2),
            value,
            font=face,
            fill=fill,
            spacing=6,
            align="center",
        )

    def arrow(x1, y1, x2, y2, color=line, stroke_width=5):
        draw.line((x1, y1, x2, y2), fill=color, width=stroke_width)
        points = [(x2, y2), (x2 - 18, y2 - 10), (x2 - 18, y2 + 10)]
        draw.polygon(points, fill=color)

    def bullet(x, y, color, value, face=f_body):
        draw.ellipse((x, y + 8, x + 9, y + 17), fill=color)
        text(x + 20, y, value, face, slate)

    text(70, 45, "最终检索评价数据集结构图", f_title, navy)
    text(
        73,
        112,
        "HotpotQA-derived final benchmark | 180 questions | 3500 documents | seed = 42 | Frozen",
        f_subtitle,
        muted,
    )
    rounded_rect((1555, 48, 1848, 126), 18, "#EEF4FF", "#C8D9FF", 2)
    center_text(
        (1555, 48, 1848, 126),
        "FINAL DATASET\n禁止在 final 上调参",
        font(23, bold=True),
        blue,
    )

    pipeline_y = 180
    card_w, card_h = 270, 128
    xs = [80, 410, 740, 1070, 1400]
    labels = [
        ("1 数据来源", "HotpotQA\ndistractor / train"),
        ("2 规则筛选", "非 yes/no\n双支撑文档"),
        ("3 候选池", "270 candidates\n90 + 90 + 90"),
        ("4 LLM 质检", "改写 / 核验 / 筛除\n只收 accepted"),
        ("5 冻结数据", "180 questions\n3500 documents"),
    ]
    colors = [soft_blue, soft_teal, soft_orange, soft_green, "#F0F4FA"]
    for index, (x, (title, body)) in enumerate(zip(xs, labels)):
        rounded_rect((x, pipeline_y, x + card_w, pipeline_y + card_h), 20, colors[index])
        text(x + 24, pipeline_y + 22, title, f_card_title, navy)
        text(x + 24, pipeline_y + 63, body, f_body, slate)
        if index < len(xs) - 1:
            arrow(x + card_w + 18, pipeline_y + 64, xs[index + 1] - 22, pipeline_y + 64)

    rounded_rect((80, 360, 675, 750), 22)
    text(112, 388, "问题集：180 个最终问题", f_section, navy)
    text(112, 430, "所有问题 split 均为 final；不再划分 dev/test。", f_small, muted)

    bar_x, bar_y = 120, 490
    bar_w, bar_h = 500, 42
    segments = [
        ("semantic_fact", task_counts["semantic_fact"], blue, "语义事实 60"),
        ("multi_hop_relation", task_counts["multi_hop_relation"], teal, "多跳关系 60"),
        ("exact_file_lookup", task_counts["exact_file_lookup"], orange, "精确/文件检索 60"),
    ]
    start = bar_x
    for _, count, color, label in segments:
        segment_w = int(bar_w * count / 180)
        draw.rounded_rectangle(
            (start, bar_y, start + segment_w, bar_y + bar_h),
            radius=14,
            fill=color,
        )
        center_text(
            (start, bar_y, start + segment_w, bar_y + bar_h),
            label,
            font(17, bold=True),
            white,
        )
        start += segment_w

    mini_y = 570
    mini_w = 162
    for index, (name, count, color, _) in enumerate(segments):
        x = 120 + index * 172
        rounded_rect((x, mini_y, x + mini_w, mini_y + 105), 16, "#F8FBFF")
        center_text((x, mini_y + 13, x + mini_w, mini_y + 53), str(count), f_num_small, color)
        center_text((x + 8, mini_y + 58, x + mini_w - 8, mini_y + 98), name, f_tiny, slate)

    text(112, 700, "exact_file_lookup 子类型：", f_card_title, navy)
    sub_x = 330
    for index, subtype in enumerate(("title_anchor", "date_number_lookup", "exact_phrase_lookup")):
        x = sub_x + index * 110
        color = [blue, green, orange][index]
        draw.rounded_rectangle((x, 696, x + 86, 728), radius=12, fill=color)
        center_text((x, 696, x + 86, 728), str(exact_counts[subtype]), font(18, bold=True), white)
    text(330, 735, "标题锚点        日期/数字        精确短语", f_tiny, muted)

    rounded_rect((715, 360, 1305, 750), 22)
    text(747, 388, "文档库：3500 篇冻结语料", f_section, navy)
    text(747, 430, "正式检索只能从 corpus.jsonl 中排序检索。", f_small, muted)

    cx, cy, cw, ch = 775, 500, 470, 64
    gold = role_counts["selected_gold"]
    noise = role_counts["noise"]
    gold_w = max(24, int(cw * gold / 3500))
    draw.rounded_rectangle((cx, cy, cx + cw, cy + ch), radius=18, fill="#EDF2FA")
    draw.rounded_rectangle((cx, cy, cx + gold_w, cy + ch), radius=18, fill=green)
    draw.rectangle((cx + gold_w - 12, cy, cx + gold_w + 6, cy + ch), fill=green)
    draw.rounded_rectangle((cx + gold_w, cy, cx + cw, cy + ch), radius=18, fill="#9AA8BD")
    center_text((cx + gold_w, cy, cx + cw, cy + ch), f"noise {noise}", font(21, bold=True), white)
    draw.line((cx + gold_w / 2, cy - 8, cx + 145, cy - 18), fill=green, width=3)
    rounded_rect((cx + 130, cy - 46, cx + 355, cy - 8), 14, soft_green, "#BFE6C8", 2)
    center_text((cx + 130, cy - 46, cx + 355, cy - 8), f"selected_gold {gold}", font(19, bold=True), green)

    bullet(765, 610, green, "240 篇被最终问题引用为 gold document")
    bullet(765, 646, "#9AA8BD", "3260 篇为噪声/干扰文档")
    bullet(765, 682, blue, "评价只看 questions.jsonl 的 gold_documents")

    rounded_rect((1345, 360, 1840, 750), 22, "#FFFDF8", "#F3D6B1", 2)
    text(1377, 388, "防污染使用边界", f_section, red)
    text(1377, 430, "运行方法与评价必须隔离。", f_small, muted)
    rules = [
        ("运行阶段", "只读 question + corpus.jsonl"),
        ("禁止读取", "gold_documents / gold_sentences / manual CSV"),
        ("评价阶段", "才读取 gold 计算 Complete@k"),
        ("禁止调参", "final 结果不得反向改 Router/Judge/参数"),
    ]
    for index, (heading, body) in enumerate(rules):
        y = 485 + index * 58
        warning = index in (1, 3)
        rounded_rect(
            (1380, y, 1802, y + 44),
            14,
            soft_red if warning else "#F8FAFD",
            "#E5D8D0" if warning else "#D8E2F0",
            1,
        )
        text(1400, y + 9, heading, font(18, bold=True), red if warning else blue)
        text(1515, y + 9, body, f_small, slate)

    rounded_rect((80, 790, 1840, 985), 22)
    text(112, 818, "后续实验读取方式", f_section, navy)
    file_cards = [
        ("输入文档库", "data/final/corpus.jsonl", "3500 docs", blue),
        ("输入问题", "data/final/questions.jsonl", "question 字段", teal),
        ("正式评价", "Complete@1 / @3 / @5", "gold 文档", orange),
        ("冻结记录", "FROZEN.txt + selection_manifest.json", "hash 可复验", green),
    ]
    fx = 115
    for title, path, info, color in file_cards:
        rounded_rect((fx, 870, fx + 405, 950), 16, "#F8FBFF")
        draw.rounded_rectangle((fx, 870, fx + 10, 950), radius=5, fill=color)
        text(fx + 25, 884, title, font(20, bold=True), navy)
        text(fx + 25, 912, path, f_tiny, slate)
        text(fx + 382, 884, info, font(17, bold=True), color, anchor="ra")
        fx += 430

    text(
        80,
        1018,
        "数据集状态：已冻结。后续方法只能在冻结语料上检索，不能用 final 结果继续调参。",
        f_small,
        muted,
    )
    text(
        1840,
        1018,
        f"questions hash: {manifest['output_hashes']['questions'][:12]}...  corpus hash: {manifest['output_hashes']['corpus'][:12]}...",
        f_tiny,
        muted,
        anchor="ra",
    )

    image.save(PNG_OUT, "PNG")


def build_svg(task_counts: Counter[str], role_counts: Counter[str]) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">
  <rect width="1920" height="1080" fill="#F7F9FC"/>
  <text x="70" y="92" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="54" font-weight="700" fill="#17233B">最终检索评价数据集结构图</text>
  <text x="73" y="143" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="26" fill="#68758B">HotpotQA-derived final benchmark | 180 questions | 3500 documents | seed = 42 | Frozen</text>
  <rect x="80" y="185" width="1760" height="120" rx="22" fill="#FFFFFF" stroke="#D8E2F0" stroke-width="2"/>
  <text x="120" y="237" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="28" font-weight="700" fill="#2F6FED">HotpotQA distractor/train</text>
  <text x="478" y="237" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="28" font-weight="700" fill="#1F9A8A">规则筛选</text>
  <text x="770" y="237" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="28" font-weight="700" fill="#F08C00">270 候选</text>
  <text x="1060" y="237" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="28" font-weight="700" fill="#2F9E44">LLM 改写/核验</text>
  <text x="1440" y="237" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="28" font-weight="700" fill="#17233B">180 题 + 3500 文档</text>
  <text x="100" y="390" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="34" font-weight="700" fill="#17233B">问题配比</text>
  <text x="120" y="450" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="26" fill="#2F6FED">semantic_fact: {task_counts['semantic_fact']}</text>
  <text x="120" y="495" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="26" fill="#1F9A8A">multi_hop_relation: {task_counts['multi_hop_relation']}</text>
  <text x="120" y="540" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="26" fill="#F08C00">exact_file_lookup: {task_counts['exact_file_lookup']}（20/20/20）</text>
  <text x="730" y="390" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="34" font-weight="700" fill="#17233B">文档库配比</text>
  <text x="750" y="450" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="26" fill="#2F9E44">selected_gold: {role_counts['selected_gold']}</text>
  <text x="750" y="495" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="26" fill="#68758B">noise: {role_counts['noise']}</text>
  <text x="750" y="540" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="26" fill="#40516B">正式检索只在 corpus.jsonl 的 3500 篇文档中进行</text>
  <text x="1280" y="390" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="34" font-weight="700" fill="#D9480F">防污染规则</text>
  <text x="1300" y="450" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="25" fill="#40516B">运行阶段只读 question + corpus</text>
  <text x="1300" y="495" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="25" fill="#40516B">评价阶段才读取 gold_documents</text>
  <text x="1300" y="540" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="25" fill="#40516B">final 结果不得反向调参</text>
  <text x="100" y="835" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="32" font-weight="700" fill="#17233B">评价口径：Complete@1 / Complete@3 / Complete@5</text>
  <text x="100" y="890" font-family="Microsoft YaHei, SimHei, sans-serif" font-size="25" fill="#68758B">正确性标签是 gold_documents / gold_sentences，不使用 metadata.answer 作为检索或路由依据。</text>
</svg>"""
    SVG_OUT.write_text(svg, encoding="utf-8")


def main() -> int:
    task_counts, exact_counts, role_counts, manifest = load_stats()
    build_png(task_counts, exact_counts, role_counts, manifest)
    build_svg(task_counts, role_counts)
    print(PNG_OUT)
    print(SVG_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
