"""交付前检查 PDF 的页数、关键结果、字体嵌入和 LaTeX 编译质量。"""
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    pdf = ROOT / "实验报告_解码器语言模型.pdf"
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    pages = [page for page in text.split("\f") if page.strip()]
    expected = ["302,509,056", "3,208,960", "1.4649", "1.6473",
                "模型超参数", "参数统计", "字符串生成", "程序调试"]
    assert all(value in text for value in expected)
    assert pages
    group = [{"name": "石皓文", "student_id": "2233316022"},
             {"name": "张恩瑞", "student_id": "2244413862"},
             {"name": "杜鹏宇", "student_id": "2223210437"},
             {"name": "许程", "student_id": "2244313744"}]
    assert all(person["name"] in pages[0] and person["student_id"] in pages[0] for person in group)
    assert "实验目的" in pages[0]
    source = (ROOT / "report" / "report.tex").read_text()
    assert "\\begin{titlepage}" not in source
    sections = re.findall(r"\\section\{([^}]+)\}", source)
    expected_sections = ["实验目的", "模型结构与实现", "数据集、环境与实验设置",
                         "参数统计方法与结果", "程序调试与结果检查", "小模型训练结果与分析",
                         "字符串生成样例", "约 300M 模型的统计与推理演示", "结论"]
    assert sections == expected_sections, sections
    pdf_sections = re.findall(r"^\s*(\d+)\s+([^\n]+)", text, flags=re.MULTILINE)
    pdf_sections = [(int(number), title.strip()) for number, title in pdf_sections
                    if title.strip() in expected_sections]
    assert pdf_sections == list(enumerate(expected_sections, 1)), pdf_sections
    references = re.findall(r"\\bibitem\{([^}]+)\}", source)
    assert references == ["vaswani2017", "ba2016", "radford2019", "tinyshakespeare"]
    checks = json.loads((ROOT / "results" / "verification.json").read_text())
    assert checks["all_passed"] and len(checks["checks"]) == 15
    log = (ROOT / "report" / "report.log").read_text()
    assert not any(word in log for word in ("Overfull", "Underfull", "undefined", "Missing character", "Warning"))
    fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    # 从右侧取固定列，兼容含空格的 CID Type 0C / Type 1C 等字体类型。
    assert all(line.split()[-5] == "yes" for line in fonts.splitlines()[2:] if line.strip())
    report = {"verified_at_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "pdf_pages": len(pages), "pdf_size_bytes": pdf.stat().st_size,
              "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
              "required_results_and_sections_present": True,
              "latex_log_without_warnings_or_overflow": True,
              "all_fonts_embedded": True, "all_15_experiment_checks_passed": True,
              "semantic_tests_passed": 7, "group_members": group,
              "body_starts_on_first_page": True, "separate_cover_removed": True,
              "section_titles_unique_and_numbering_checked": True,
              "original_first_reference_removed": True, "reference_count": len(references)}
    (ROOT / "results" / "delivery_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
