
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Diagram import config
from src.Diagram.pipeline import preview_image, run, run_preview


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="试纸卡盒九宫格判读: 配准 -> 质控 -> E 集成 -> Grad-CAM -> 结果图"
    )
    parser.add_argument("--image", type=str, default="",
                        help="待判读的新照片路径; 留空则只输出参考图 ROI 预览")
    parser.add_argument("--preview", action="store_true",
                        help="只输出参考图上的 9 框预览 (确认坐标), 不做判读")
    parser.add_argument("--preview-image", action="store_true",
                        help="配合 --image: 只输出该图的配准/孔位/裁剪预览, 不跑模型")
    parser.add_argument("--no-gradcam", action="store_true", help="跳过 Grad-CAM 热力图")
    parser.add_argument("--threshold", type=float, default=None,
                        help="主阈值覆盖; 缺省用各 E 模型 metrics.csv best_threshold 中位数")
    parser.add_argument("--device", type=str, default=None, help="推理设备: cuda / cpu")
    parser.add_argument("--force", action="store_true",
                        help="强制重新标定参考图并重建 ROI 配置")
    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.preview or (not args.image and not args.preview_image):
        out = run_preview(force=args.force)
        print("\n参考图 ROI 预览已生成 (请先核对坐标是否合理):")
        print(f"  标准矩形俯视图: {out}")
        print(f"  原图反投影:     {config.PREVIEW_DIR / 'roi_preview_original.png'}")
        if not args.image:
            print("\n确认无误后, 用 --image <照片路径> 跑完整判读流程。")
        return

    if args.preview_image:
        info = preview_image(args.image)
        print("\n" + "#" * 70)
        print("# 配准 / 孔位 / 裁剪预览")
        print("#" * 70)
        print(f"配准方法: {info['method']}")
        print(f"孔位来源: {info['source']}")
        grid = info.get("grid")
        if grid and grid.get("columns"):
            print("  列范围 (标准矩形内像素): "
                  + ", ".join(f"{x0:.0f}-{x1:.0f}" for x0, x1 in grid["columns"]))
            print("  行范围: " + ", ".join(f"{y:.0f}" for y in grid["rows"]))
        print("-" * 70)
        print(f"标准矩形 + 9 框: {info['standard']}")
        print(f"原图反投影:      {info['original']}")
        print(f"裁剪拼图 (模型输入): {info['crops']}")
        return

    payload = run(
        args.image,
        device=args.device,
        threshold=args.threshold,
        use_gradcam=not args.no_gradcam,
    )

    print("\n" + "#" * 70)
    print("# 判读汇总")
    print("#" * 70)
    qc = payload["qc"]
    print(f"卡盒检测: {payload['card_detect_method']} | "
          f"质控: {'有效' if qc['valid'] else '无效'} (波峰数={qc['n_peaks']})")
    print(f"阈值: 主={payload['threshold']['primary']:.4f} "
          f"(阴性<{payload['threshold']['neg_max']:.4f}, 阳性>{payload['threshold']['pos_min']:.4f})")
    print("-" * 70)
    for tag in config.TEST_TAGS:
        item = payload["results"][tag]
        print(f"  {tag:<3} {item['name']:<9} prob={item['prob']:.4f}  ->  {item['label']}")
    print("-" * 70)
    print(payload["summary"])
    print("-" * 70)
    print(f"结果图: {payload['outputs']['result_image']}")
    if payload["outputs"]["heatmaps"]:
        print(f"热力图: {config.HEATMAPS_DIR / Path(args.image).stem} ({len(payload['outputs']['heatmaps'])} 张)")
    print(f"完整结果 JSON: {Path(payload['outputs']['result_image']).parent / 'result.json'}")


if __name__ == "__main__":
    main()
