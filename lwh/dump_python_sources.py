#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
递归导出目录下所有 Python 文件内容到一个 txt 文件

输出格式：
========================================
文件路径: <absolute_or_relative_path>
文件名: <filename.py>
----------------------------------------
<file content>
========================================
"""

import os
from pathlib import Path


def dump_python_files(
    root_dir: Path,
    out_txt: Path,
    encoding: str = "utf-8"
):
    root_dir = root_dir.resolve()

    with out_txt.open("w", encoding="utf-8") as fout:
        for py_file in sorted(root_dir.rglob("*.py")):
            # 可选：跳过虚拟环境 / 缓存目录
            if any(part in {"__pycache__", ".venv", "venv", "env"} for part in py_file.parts):
                continue

            fout.write("=" * 80 + "\n")
            fout.write(f"文件路径: {py_file.parent}\n")
            fout.write(f"文件名: {py_file.name}\n")
            fout.write("-" * 80 + "\n")

            try:
                content = py_file.read_text(encoding=encoding)
            except UnicodeDecodeError:
                # 如果不是 utf-8，尝试忽略错误读取
                content = py_file.read_text(encoding=encoding, errors="ignore")

            fout.write(content)
            if not content.endswith("\n"):
                fout.write("\n")

            fout.write("=" * 80 + "\n\n")


if __name__ == "__main__":
    # ===== 你只需要改这里 =====
    ROOT_DIR = Path(".")              # 要遍历的目录（当前目录）
    OUTPUT_TXT = Path("all_py_dump.txt")  # 输出文件名

    dump_python_files(ROOT_DIR, OUTPUT_TXT)
    print(f"[OK] Python 源码已导出到: {OUTPUT_TXT.resolve()}")
