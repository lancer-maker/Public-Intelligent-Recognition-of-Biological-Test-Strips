
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from new_env_train import prepare_labels, read_csv_directly
from src.data.dataset import TestStripDataset
from src.data.transforms import get_val_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state

CONFIG_PATH = PROJECT_ROOT / "configs" / "main_config.yaml"
EVAL_CSV = "data/test/labels/labels.csv"
EVAL_RAW_DIR = PROJECT_ROOT / "data" / "test" / "raw"
MODEL_IDS: List[int] = [1, 2, 3, 4, 5]
OUT_CSV = PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "proj_sensitivity.csv"
SEED = 0
rng = np.random.default_rng(SEED)


def _load_config() -> dict:

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> DualStreamStripNet:

    return DualStreamStripNet(
        pretrained_2d=bool(cfg["model"]["pretrained"]),
        feature_dim=int(cfg["model"]["feature_dim"]),
        dropout_rate=float(cfg["model"]["dropout_rate"]),
    )


def build_eval_dataset(cfg: dict):

    df = read_csv_directly(str(EVAL_CSV))
    prepared = prepare_labels(df, EVAL_RAW_DIR)
    ds = TestStripDataset(
        image_paths=prepared["filename"].tolist(),
        labels=prepared["label"].tolist(),
        transforms=get_val_transforms(),
        projection_transforms=False,
        standard_dpi_size=tuple(cfg["data"]["standard_dpi_size"]),
        proj_length=int(cfg["data"]["proj_length"]),
    )
    return ds, prepared["label"].astype(int).to_numpy()


def build_variants(x: torch.Tensor):

    variants = {"baseline": x.clone()}
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)

    for c in (0.05, 0.15, -0.15):
        variants[f"dc{c:+.2f}"] = x + c

    for g in (0.3, 0.6, 1.5, 3.0):
        variants[f"gain{g:.1f}"] = x * g

    variants["shape_only"] = x / std

    variants["flip"] = torch.flip(x, dims=[-1])

    n_pts = x.shape[-1]
    n_blocks = 8
    blk = n_pts // n_blocks
    order = rng.permutation(n_blocks)
    x_shuf = torch.empty_like(x)
    for ch in range(x.shape[0]):
        blocks = [x[ch, i * blk:(i + 1) * blk].clone() for i in range(n_blocks)]
        x_shuf[ch] = torch.cat([blocks[o] for o in order], dim=0)
    variants["block_shuffle"] = x_shuf

    torch.manual_seed(SEED)
    variants["noise_0.02"] = x + torch.randn_like(x) * 0.02

    return variants


def main():
    cfg = _load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 配置: {CONFIG_PATH.name}")

    ds, labels = build_eval_dataset(cfg)
    names = [Path(p).name for p in ds.image_paths]
    print(f"分析数据集: {len(names)} 张 (阳性 {int(labels.sum())})")

    rows = []
    for mid in MODEL_IDS:
        ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "Tweak" / str(mid) / "best_model.pth"
        if not ckpt_path.exists():
            print(f"[跳过] 无权重: {ckpt_path}")
            continue
        model = build_model(cfg).to(device)
        model = load_model_state(model, str(ckpt_path))
        model.eval()

        per_cond = {k: [] for k in
                    ["baseline", "dc+0.05", "dc+0.15", "dc-0.15",
                     "gain0.3", "gain0.6", "gain1.5", "gain3.0",
                     "shape_only", "flip", "block_shuffle", "noise_0.02"]}
        accs = {k: [] for k in per_cond}
        proj_only_base, image_only_base, fusion_base = [], [], []

        with torch.no_grad():
            for i in range(len(ds)):
                y = int(labels[i])
                img_t, proj_t, _ = ds[i]
                img_t = img_t.unsqueeze(0).to(device)
                x0 = proj_t.to(device)
                img_feats = model.image_stream(img_t)
                zero = torch.zeros_like(img_feats)

                def prob_fusion(x):
                    pf = model.proj_stream(x.unsqueeze(0))
                    return torch.sigmoid(model.classifier(
                        torch.cat([img_feats, pf], dim=1))).item()

                def prob_proj(x):
                    pf = model.proj_stream(x.unsqueeze(0))
                    return torch.sigmoid(model.classifier(
                        torch.cat([zero, pf], dim=1))).item()

                proj_only_base.append(prob_proj(x0))
                image_only_base.append(torch.sigmoid(model.classifier(
                    torch.cat([img_feats, zero], dim=1))).item())
                fusion_base.append(prob_fusion(x0))

                for name, xv in build_variants(x0).items():
                    p = prob_fusion(xv)
                    per_cond[name].append(p)
                    accs[name].append(int(p >= 0.5) == y)

        n = len(fusion_base)
        base_fusion = np.asarray(fusion_base)
        base_acc = np.mean(np.asarray(accs["baseline"]))
        ponly = np.asarray(proj_only_base)
        ionly = np.asarray(image_only_base)

        print(f"\n=== 模型 {mid} ===")
        print(f"  fusion_acc={base_acc:.3f} | image_only_acc="
              f"{np.mean((ionly>=0.5)==labels):.3f} | proj_only_acc="
              f"{np.mean((ponly>=0.5)==labels):.3f}")

        row_base = {
            "model": f"Tweak/{mid}", "condition": "baseline",
            "fusion_acc": round(base_acc, 4),
            "proj_only_acc": round(float(np.mean((ponly >= 0.5) == labels)), 4),
            "image_only_acc": round(float(np.mean((ionly >= 0.5) == labels)), 4),
            "mean_abs_delta": 0.0, "flip_rate": 0.0,
        }
        rows.append(row_base)

        for cond in per_cond:
            if cond == "baseline":
                continue
            arr = np.asarray(per_cond[cond])
            mean_delta = float(np.mean(np.abs(arr - base_fusion)))
            flip_rate = float(np.mean((arr >= 0.5) != (base_fusion >= 0.5)))
            acc = float(np.mean(accs[cond]))
            print(f"  {cond:<14s} mean|Δ|={mean_delta:.4f}  flip={flip_rate:.3f}  acc={acc:.3f}")
            rows.append({
                "model": f"Tweak/{mid}", "condition": cond,
                "fusion_acc": round(acc, 4),
                "proj_only_acc": None, "image_only_acc": None,
                "mean_abs_delta": round(mean_delta, 4),
                "flip_rate": round(flip_rate, 4),
            })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {OUT_CSV}")


if __name__ == "__main__":
    main()
