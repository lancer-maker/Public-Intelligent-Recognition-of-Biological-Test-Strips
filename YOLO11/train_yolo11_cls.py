

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yolo11_common import (
    EXPECTED_COUNTS,
    REPORTS_DIR,
    RUNS_DIR,
    SEEDS,
    dataset_root_for_training,
    dump_json,
    ensure_output_dirs,
    environment_fingerprint,
    resolve_pretrained_weights,
    tee_stdout,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="YOLO11 分类模式多随机种子训练")
    parser.add_argument("--model", default="yolo11n-cls.pt",
                        help="模型/权重: 名称 (如 yolo11n-cls.pt / yolo11s-cls.pt) 或本地权重路径")
    parser.add_argument("--imgsz", type=int, default=224, choices=[224, 320, 256, 384, 448, 512],
                        help="输入尺寸 (YOLO11 分类默认 224, 推荐 224 或 320)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (分类任务默认 100)")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值 (验证指标不提升的轮数)")
    parser.add_argument("--optimizer", default="auto",
                        help="优化器: auto / SGD / Adam / AdamW (auto 时分类任务通常选 AdamW)")
    parser.add_argument("--lr0", type=float, default=None,
                        help="初始学习率; 留空则使用 Ultralytics 自动策略")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"随机种子列表 (默认 {SEEDS})")
    parser.add_argument("--device", default="0", help="设备: 0 / 0,1 / cpu")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader 线程数")
    parser.add_argument("--cache", default="False",
                        choices=["False", "True", "ram", "disk"],
                        help="图像缓存方式 (数据集很小, ram 可显著加速)")
    parser.add_argument("--tag", default=None,
                        help="运行分组名 (默认由模型名与输入尺寸生成, 例如 yolo11n-cls_224)")
    parser.add_argument("--skip-prepare", action="store_true", help="跳过数据集校验信息打印")
    parser.add_argument("--force-prepare", action="store_true",
                        help="归一化复制模式时强制重新复制图像 (零拷贝模式无影响)")
    return parser.parse_args()


def default_tag(model: str, imgsz: int) -> str:

    stem = Path(model).stem or "yolo11-cls"
    return f"{stem}_{imgsz}"


def train_one_seed(seed: int, weights: Path, args: argparse.Namespace, tag: str,
                   data_root: Path) -> Path:

    from ultralytics import YOLO

    run_project = RUNS_DIR / tag
    run_name = f"seed_{seed}"
    run_dir = run_project / run_name
    log_path = REPORTS_DIR / "logs" / f"train_{tag}_seed{seed}.log"

    print("\n" + "=" * 78)
    print(f"[train] seed={seed}  权重={weights}  输出={run_dir}")
    print(f"[train] 数据集根目录: {data_root}")
    print("=" * 78)

    train_kwargs: Dict[str, object] = {
        "data": str(data_root),
        "imgsz": int(args.imgsz),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "patience": int(args.patience),
        "optimizer": args.optimizer,
        "pretrained": True,
        "seed": int(seed),
        "deterministic": True,
        "project": str(run_project),
        "name": run_name,
        "exist_ok": True,
        "device": args.device,
        "workers": int(args.workers),
        "cache": args.cache,
        "plots": True,
        "verbose": True,
    }
    if args.lr0 is not None:
        train_kwargs["lr0"] = float(args.lr0)

    with tee_stdout(log_path):
        model = YOLO(str(weights))
        model.train(**train_kwargs)

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"训练结束但未找到 best.pt: {best}")
    print(f"[train] seed={seed} 完成 -> {best}")
    return best


def main() -> None:

    args = parse_args()
    ensure_output_dirs()

    data_root = dataset_root_for_training(verbose=not args.skip_prepare, force=args.force_prepare)

    tag = args.tag or default_tag(args.model, args.imgsz)
    weights = resolve_pretrained_weights(args.model)

    config = {
        "task": "classify",
        "model": args.model,
        "pretrained_weights": str(weights),
        "dataset_dir": str(data_root),
        "expected_counts": EXPECTED_COUNTS,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "seeds": list(args.seeds),
        "tag": tag,
        "device": args.device,
        "workers": args.workers,
        "cache": args.cache,
        "deterministic": True,
        "environment": environment_fingerprint(),
    }
    dump_json(REPORTS_DIR / f"train_config_{tag}.json", config)

    print("训练配置:")
    for key in ("model", "imgsz", "epochs", "batch", "patience", "optimizer", "tag", "device"):
        print(f"  {key}: {config[key]}")
    print(f"  随机种子: {config['seeds']}")

    finished: List[Dict[str, object]] = []
    for seed in args.seeds:
        best = train_one_seed(seed, weights, args, tag, data_root)
        finished.append({"seed": int(seed), "best_pt": str(best)})

    dump_json(REPORTS_DIR / f"train_runs_{tag}.json", {"tag": tag, "runs": finished})
    print("\n全部种子训练完成:")
    for item in finished:
        print(f"  seed={item['seed']:<5} -> {item['best_pt']}")
    print("下一步: python YOLO11/test_yolo11_cls.py --tag " + tag)


if __name__ == "__main__":
    main()
