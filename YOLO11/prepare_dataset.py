

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import TEST_DIR, TRAIN_DIR, VAL_DIR
from yolo11_common import (
    DATASET_DIR,
    REPORTS_DIR,
    dataset_root_for_training,
    dataset_summary,
    ensure_output_dirs,
    environment_fingerprint,
    write_csv,
)

SUMMARY_COLUMNS = ["split", "configured", "path", "n", "n_neg", "n_pos", "expected", "match"]


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="校验/准备 YOLO11 分类数据集")
    parser.add_argument("--force", action="store_true",
                        help="归一化复制时强制覆盖已存在文件 (零拷贝模式无影响)")
    parser.add_argument("--verify-only", action="store_true",
                        help="只校验配置目录, 不做任何复制")
    return parser.parse_args()


def main() -> None:

    args = parse_args()
    ensure_output_dirs()

    print("=" * 78)
    print("YOLO11 分类数据集校验")
    for key, value in environment_fingerprint().items():
        print(f"  {key}: {value}")
    print("-" * 78)
    for name, value in (("TRAIN_DIR", TRAIN_DIR), ("VAL_DIR", VAL_DIR), ("TEST_DIR", TEST_DIR)):
        shown = value if str(value).strip() else "<未配置, 请填写 YOLO11/dataset_config.py>"
        print(f"  {name:<9}: {shown}")
    print("=" * 78)

    summary = dataset_summary()
    df = pd.DataFrame(summary)[SUMMARY_COLUMNS]
    out_csv = REPORTS_DIR / "dataset_summary.csv"
    write_csv(df, out_csv)
    print(df.to_string(index=False))

    missing = [row for row in summary if not row["configured"]]
    if missing:
        print("-" * 78)
        for row in missing:
            print(f"[未配置] {row['split']}: {row['message']}")
        raise SystemExit("数据集尚未配置完成, 请先编辑 YOLO11/dataset_config.py 填入三个目录。")

    print("-" * 78)
    for row in summary:
        flag = "OK" if row["match"] else f"期望 {row['expected']}, 实际 {row['n']} -> 不一致"
        print(f"  {row['split']:<5} {row['n']:>4} 张 (N={row['n_neg']}, P={row['n_pos']})  {flag}")
    print(f"统计已保存: {out_csv}")

    if args.verify_only:
        print("仅校验模式: 未复制任何文件。")
        return

    root = dataset_root_for_training(verbose=True, force=args.force)
    if root == DATASET_DIR:
        print(f"已归一化复制到: {DATASET_DIR}")
    else:
        print(f"训练将直接使用用户数据根目录 (零拷贝): {root}")
    print("数据集就绪, 可以开始训练。")


if __name__ == "__main__":
    main()
