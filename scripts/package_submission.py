"""将源码、数据、权重、日志和 LaTeX/PDF 打成可独立审阅的提交包。"""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "编程作业二_完整提交包.zip"


def main():
    required = ["实验报告_解码器语言模型.pdf", "report/report.tex", "src/model.py",
                "results/small_final.pt", "results/large_summary.json", "results/verification.json"]
    for name in required:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
    if not json.loads((ROOT / "results/verification.json").read_text())["all_passed"]:
        raise ValueError("验证未通过")
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == DESTINATION or path.name == "MANIFEST.sha256":
            continue
        if any(x in path.parts for x in ("__pycache__", ".pytest_cache", "preview")):
            continue
        if path.suffix in (".aux", ".toc", ".out", ".pyc"):
            continue
        files.append(path)
    manifest = "\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}" for p in files) + "\n"
    (ROOT / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
    with zipfile.ZipFile(DESTINATION, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files + [ROOT / "MANIFEST.sha256"]:
            archive.write(path, ROOT.name + "/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(DESTINATION) as archive:
        assert archive.testzip() is None
    print(f"Created: {DESTINATION} ({DESTINATION.stat().st_size / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
