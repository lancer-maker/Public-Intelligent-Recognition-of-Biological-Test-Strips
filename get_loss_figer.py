
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.figure_result import PLOT_DPI, apply_global_style

MODEL_DIR = r"outputs\checkpoints\C\2"

COLOR_TRAIN = "#1f77b4"
COLOR_SMOOTH = "#ff7f0e"
COLOR_VAL = "#d62728"

STYLE_TRAIN = "-"
STYLE_SMOOTH = "--"
STYLE_VAL = ":"
LINE_WIDTH = 1.8

SMOOTH_WINDOW = 15
FIG_SIZE = (8.0, 5.5)
XLABEL = "Epoch"
YLABEL = "Loss"
SHOW_GRID = True
TITLE_TEMPLATE = "{model} (seed {seed})"

LEGEND_LOC = "upper left"
LEGEND_ANCHOR = (1.0, 1.0)
LEGEND_BORDER_AXESPAD = 0.0
LEGEND_FRAMEON = True

CONFIG_PATH = r"configs\main_config.yaml"
ANNOTATE_STAGE = True
ANNOTATE_UNFREEZE = True
COLOR_STAGE = "#7f7f7f"
STYLE_STAGE = "--"
UNFREEZE_COLORS = ["#5b3a86", "#a97fd0", "#c9a8e0",
                   "#3f2860", "#e0cdf0"]
STYLE_UNFREEZE = "-."
ANNOT_LINE_WIDTH = 1.2
ANNOT_LINE_ALPHA = 0.9
STAGE_LABEL = "Stage1 | Stage2"
ANNOT_POS_IN_LABEL = False
BACKBONE_NUM_BLOCKS = 7

OUT_FULL = "loss_curve_full.svg"
OUT_TRAIN = "loss_curve_train.svg"

LineSpec = Tuple[Sequence[float], str, str, str, float]


def resolve_model_dir(raw: str) -> Path:

    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def model_name_from_dir(model_dir: Path) -> str:

    for parent in [model_dir] + list(model_dir.parents):
        name = parent.name
        if not name or name.isdigit():
            continue
        if name.lower() in {"outputs", "checkpoints", "reports", "ensemble_report"}:
            continue
        return name
    return model_dir.name


def moving_average(values: Sequence[float], window: int = 15) -> np.ndarray:

    values = np.asarray(values, dtype=float)
    if window is None or window <= 1 or len(values) == 0:
        return values.copy()
    window = min(int(window), len(values))
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_history(model_dir: Path) -> pd.DataFrame:

    csv_path = model_dir / "history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到训练历史文件: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = {"train_loss", "val_loss"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} 缺少必需列: {sorted(missing)}; 实际列为: {list(df.columns)}")
    if len(df) == 0:
        raise ValueError(f"{csv_path} 为空。")
    return df


def load_config_dict(raw_path: str) -> dict:

    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"[警告] 配置文件不存在, 跳过解冻标注: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[警告] 读取配置失败, 跳过解冻标注: {exc}")
        return {}


def stage2_start_row(df: pd.DataFrame) -> Optional[int]:

    if "stage" not in df.columns:
        return None
    stages = df["stage"].astype(str).str.strip().str.lower().to_numpy()
    idx = np.flatnonzero(stages == "stage2")
    return int(idx[0]) if idx.size else None


def build_unfreeze_order(start_block, num_layers, n_blocks: int) -> List[str]:

    n = max(1, int(n_blocks))
    start = n - 1 if start_block is None else int(start_block)
    start = max(0, min(n - 1, start))
    num = start + 1 if num_layers is None else max(1, int(num_layers))
    num = min(num, start + 1)
    return [f"blocks.{i}" for i in range(start, start - num, -1)]


def resolve_annotations(df: pd.DataFrame,
                        cfg: dict) -> Tuple[Optional[float], List[Tuple[float, str]]]:

    train_cfg = (cfg or {}).get("training", {}) or {}
    s1_cfg = train_cfg.get("stage1", {}) or {}
    s2_cfg = train_cfg.get("stage2", {}) or {}

    s2_row = stage2_start_row(df)
    if s2_row is not None:
        stage2_rows = len(df) - s2_row
        stage2_first_x = s2_row + 1
        boundary: Optional[float] = s2_row + 0.5
    else:
        s1_epochs = int(s1_cfg.get("epochs", 0) or 0)
        stage2_rows = int(s2_cfg.get("epochs", 0) or 0)
        stage2_first_x = s1_epochs + 1
        boundary = (s1_epochs + 0.5) if s1_epochs > 0 else None

    interval = int(s2_cfg.get("unfreeze_interval", 0) or 0)
    marks: List[Tuple[float, str]] = []
    if interval > 0 and stage2_rows > 0:
        order = build_unfreeze_order(s2_cfg.get("unfreeze_start_block"),
                                     s2_cfg.get("unfreeze_num_layers"),
                                     BACKBONE_NUM_BLOCKS)
        for k, blk in enumerate(order):
            stage2_epoch = (k + 1) * interval
            if stage2_epoch > stage2_rows:
                break
            marks.append((float(stage2_first_x - 1 + stage2_epoch), blk))
    return boundary, marks


def plot_series(x: Sequence[float], series: List[LineSpec], out_path: Path, title: str,
                stage_boundary: Optional[float] = None,
                unfreeze_marks: Sequence[Tuple[float, str]] = ()) -> None:

    apply_global_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for y, label, color, linestyle, linewidth in series:
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth,
                label=label, zorder=3)

    if stage_boundary is not None:
        stage_tag = (f"{STAGE_LABEL} @ e{stage_boundary:g}"
                     if ANNOT_POS_IN_LABEL else STAGE_LABEL)
        ax.axvline(stage_boundary, color=COLOR_STAGE, linestyle=STYLE_STAGE,
                   linewidth=ANNOT_LINE_WIDTH, alpha=ANNOT_LINE_ALPHA, zorder=2,
                   label=stage_tag)

    uf_colors = list(UNFREEZE_COLORS) if UNFREEZE_COLORS else ["#9467bd"]
    for i, (px, blk) in enumerate(unfreeze_marks):
        block_tag = f"{blk} @ e{px:g}" if ANNOT_POS_IN_LABEL else blk
        ax.axvline(px, color=uf_colors[i % len(uf_colors)], linestyle=STYLE_UNFREEZE,
                   linewidth=ANNOT_LINE_WIDTH, alpha=ANNOT_LINE_ALPHA, zorder=2,
                   label=block_tag)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    if SHOW_GRID:
        ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc=LEGEND_LOC, bbox_to_anchor=LEGEND_ANCHOR,
              borderaxespad=LEGEND_BORDER_AXESPAD, frameon=LEGEND_FRAMEON)
    ax.set_title(title)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".svg":
        fig.savefig(out_path, format="svg", bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="训练 Loss 曲线绘制 (单文件夹, 输出两幅图)")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="模型输出文件夹 (覆盖配置区 MODEL_DIR), 每次只处理一个")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir if args.model_dir else MODEL_DIR)
    if not model_dir.exists():
        print(f"[错误] 文件夹不存在: {model_dir}")
        return

    try:
        df = load_history(model_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}")
        return

    train = df["train_loss"].to_numpy(dtype=float)
    val = df["val_loss"].to_numpy(dtype=float)
    smooth = moving_average(train, SMOOTH_WINDOW)
    x = np.arange(1, len(train) + 1)

    seed = int(df["seed"].iloc[0]) if "seed" in df.columns and not pd.isna(df["seed"].iloc[0]) else -1
    model_name = model_name_from_dir(model_dir)
    title = TITLE_TEMPLATE.format(model=model_name, seed=seed)

    smooth_label = "Train Loss (smooth)" if SMOOTH_WINDOW <= 1 \
        else f"Train Loss (smooth, w={int(SMOOTH_WINDOW)})"

    cfg = load_config_dict(CONFIG_PATH)
    stage_boundary, unfreeze_marks = resolve_annotations(df, cfg)
    if not ANNOTATE_STAGE:
        stage_boundary = None
    if not ANNOTATE_UNFREEZE:
        unfreeze_marks = []

    full_path = model_dir / OUT_FULL
    train_path = model_dir / OUT_TRAIN

    plot_series(x, [
        (train, "Train Loss", COLOR_TRAIN, STYLE_TRAIN, LINE_WIDTH),
        (smooth, smooth_label, COLOR_SMOOTH, STYLE_SMOOTH, LINE_WIDTH),
        (val, "Val Loss", COLOR_VAL, STYLE_VAL, LINE_WIDTH),
    ], full_path, f"{title} — Train/Val Loss",
        stage_boundary=stage_boundary, unfreeze_marks=unfreeze_marks)

    plot_series(x, [
        (train, "Train Loss", COLOR_TRAIN, STYLE_TRAIN, LINE_WIDTH),
        (smooth, smooth_label, COLOR_SMOOTH, STYLE_SMOOTH, LINE_WIDTH),
    ], train_path, f"{title} — Train Loss",
        stage_boundary=stage_boundary, unfreeze_marks=unfreeze_marks)

    n_val = int(np.isfinite(val).sum())
    print(f"[*] 文件夹: {model_dir}")
    print(f"[*] 模型/seed: {model_name} / {seed} | 记录 {len(train)} 个 epoch "
          f"(有效 val_loss {n_val} 个) | 平滑窗口 {SMOOTH_WINDOW}")
    if stage_boundary is not None:
        print(f"[*] Stage1|Stage2 分水岭: x={stage_boundary:.1f} (累计 epoch)")
    else:
        print("[*] 未找到 Stage2 信息, 跳过分水岭标注")
    if unfreeze_marks:
        print("[*] 解冻 block 位置: "
              + ", ".join(f"{blk}@epoch{int(round(px))}" for px, blk in unfreeze_marks))
    else:
        print("[*] 无可标注的解冻 block (检查配置或 Stage2 长度)")
    print(f"    已保存: {full_path}")
    print(f"    已保存: {train_path}")


if __name__ == "__main__":
    main()
