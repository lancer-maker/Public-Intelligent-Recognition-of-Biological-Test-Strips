
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state

CONFIG_PATH = PROJECT_ROOT / "configs" / "main_config.yaml"
MODEL_ROOT = PROJECT_ROOT / "outputs" / "checkpoints" / "Tweak"
MODEL_IDS: List[int] = [1, 2, 3, 4, 5]
OUT_CSV = PROJECT_ROOT / "outputs" / "reports" / "stream_importance.csv"


def _load_config() -> dict:

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> DualStreamStripNet:

    return DualStreamStripNet(
        pretrained_2d=bool(cfg["model"]["pretrained"]),
        feature_dim=int(cfg["model"]["feature_dim"]),
        dropout_rate=float(cfg["model"]["dropout_rate"]),
    )


def resolve_model_ids() -> List[int]:

    if MODEL_IDS:
        return list(MODEL_IDS)
    if not MODEL_ROOT.exists():
        return []
    return sorted(int(p.name) for p in MODEL_ROOT.iterdir()
                  if p.is_dir() and p.name.isdigit())


def main() -> None:
    cfg = _load_config()
    model_ids = resolve_model_ids()
    if not model_ids:
        raise FileNotFoundError(f"配置区未指定且 {MODEL_ROOT} 下没有数字模型目录")

    feature_dim = int(cfg["model"]["feature_dim"])
    rows = []
    print("=" * 70)
    print(f"分类头第一层权重 (输入 {feature_dim*2} = 图像{feature_dim} + 投影{feature_dim}) 流重要性分析")
    print("=" * 70)

    for mid in model_ids:
        ckpt_path = MODEL_ROOT / str(mid) / "best_model.pth"
        if not ckpt_path.exists():
            print(f"[跳过] 无权重: {ckpt_path}")
            continue
        model = build_model(cfg)
        model = load_model_state(model, str(ckpt_path))
        model.eval()

        fc_weight = model.classifier[0].weight.data
        img_part = fc_weight[:, :feature_dim]
        proj_part = fc_weight[:, feature_dim:]

        img_l1 = img_part.abs().sum().item()
        proj_l1 = proj_part.abs().sum().item()
        img_l2 = torch.norm(img_part).item()
        proj_l2 = torch.norm(proj_part).item()

        if proj_l1 > img_l1 * 3:
            hint = "明显更依赖投影流, 图像流可能被忽略"
        elif img_l1 > proj_l1 * 3:
            hint = "明显更依赖图像流, 投影流可能被忽略"
        else:
            hint = "两个流的权重分布相对均衡"

        print(f"\n模型 {mid}  ({ckpt_path.name})")
        print(f"  图像流 L1={img_l1:.4f}  L2={img_l2:.4f}")
        print(f"  投影流 L1={proj_l1:.4f}  L2={proj_l2:.4f}")
        print(f"  → {hint}")

        rows.append({
            "model_id": mid,
            "weight_path": str(ckpt_path),
            "img_L1": round(img_l1, 4), "img_L2": round(img_l2, 4),
            "proj_L1": round(proj_l1, 4), "proj_L2": round(proj_l2, 4),
            "hint": hint,
        })

    if OUT_CSV and rows:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n已保存: {OUT_CSV}")


if __name__ == "__main__":
    main()