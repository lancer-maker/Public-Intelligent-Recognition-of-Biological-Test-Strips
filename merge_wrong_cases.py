
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

WRONG_ROOT = PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "wrong_cases"


def main() -> None:
    if not WRONG_ROOT.exists():
        raise FileNotFoundError(f"未找到目录: {WRONG_ROOT}")

    folders = sorted([p for p in WRONG_ROOT.iterdir() if p.is_dir()])
    if not folders:
        raise FileNotFoundError(f"{WRONG_ROOT} 下无模型子文件夹")

    for folder in folders:
        model_name = folder.name
        csvs = sorted(folder.glob("*.csv"))
        if not csvs:
            print(f"[跳过] {folder} 无 csv")
            continue

        rows = []
        for f in csvs:
            if f.stem == model_name:
                continue
            df = pd.read_csv(f)
            prob_col = f"{model_name}_prob"
            col = prob_col if prob_col in df.columns else [c for c in df.columns if c.endswith("_prob")][0]
            sub = df[["sample_id", "filename", "true_label", col]].copy()
            sub = sub.rename(columns={col: prob_col})
            rows.append(sub)

        merged = pd.concat(rows, ignore_index=True)
        merged = merged.drop_duplicates(subset=["sample_id"]).sort_values("sample_id")
        merged["sample_id"] = merged["sample_id"].astype(int)

        out = folder / f"{model_name}.csv"
        merged.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[完成] {out}  (共 {len(merged)} 行, 列: {list(merged.columns)})")


if __name__ == "__main__":
    main()
