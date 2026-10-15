

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

YOLO11_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = YOLO11_DIR.parent


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="YOLO11 分类对比实验一键运行")
    parser.add_argument("--model", default="yolo11n-cls.pt", help="模型名称或权重路径")
    parser.add_argument("--imgsz", type=int, default=224, help="输入尺寸 (默认 224, 推荐 224/320)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="训练批大小")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值")
    parser.add_argument("--optimizer", default="auto", help="优化器 (auto/SGD/Adam/AdamW)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 831, 2026, 7, 2691],
                        help="随机种子列表")
    parser.add_argument("--device", default="0", help="设备: 0 / cpu")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader 线程数")
    parser.add_argument("--tag", default=None, help="运行分组名 (默认 <模型名>_<imgsz>)")
    parser.add_argument("--skip-prepare", action="store_true", help="跳过数据集构建")
    parser.add_argument("--force-prepare", action="store_true", help="构建数据集时强制重新复制")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练 (复用已有权重)")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估")
    parser.add_argument("--no-figures", action="store_true", help="评估时不出图")
    return parser.parse_args()


def run_step(title: str, cmd: List[str]) -> None:

    print("\n" + "#" * 78)
    print(f"# {title}")
    print("# " + " ".join(cmd))
    print("#" * 78)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"步骤失败 (exit={result.returncode}): {title}")


def main() -> None:

    args = parse_args()
    python = sys.executable
    tag = args.tag or f"{Path(args.model).stem}_{args.imgsz}"

    if not args.skip_prepare:
        cmd = [python, str(YOLO11_DIR / "prepare_dataset.py")]
        if args.force_prepare:
            cmd.append("--force")
        run_step("步骤 1/3: 校验 YOLO11 分类数据集", cmd)

    if not args.skip_train:
        cmd = [
            python, str(YOLO11_DIR / "train_yolo11_cls.py"),
            "--model", args.model,
            "--imgsz", str(args.imgsz),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--patience", str(args.patience),
            "--optimizer", args.optimizer,
            "--device", args.device,
            "--workers", str(args.workers),
            "--tag", tag,
            "--seeds", *[str(s) for s in args.seeds],
            "--skip-prepare",
        ]
        run_step("步骤 2/3: 多随机种子训练 YOLO11 分类模型", cmd)

    if not args.skip_eval:
        cmd = [
            python, str(YOLO11_DIR / "test_yolo11_cls.py"),
            "--tag", tag,
            "--imgsz", str(args.imgsz),
            "--device", args.device,
            "--seeds", *[str(s) for s in args.seeds],
        ]
        if args.no_figures:
            cmd.append("--no-figures")
        run_step("步骤 3/3: 验证/测试评估 + 多种子集成", cmd)

    print("\n全部完成。结果目录: outputs/YOLO11 "
          "(runs/ 训练权重, reports/ 指标与逐样本概率, figures/ 图表)")


if __name__ == "__main__":
    main()
