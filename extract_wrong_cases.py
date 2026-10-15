
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

PRED_DIR = PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "predictions"
LABELS_CSV = PROJECT_ROOT / "data" / "test" / "labels" / "labels.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "wrong_cases"


def load_pred_sources() -> list:

    files = sorted(PRED_DIR.glob("*preds.csv"), key=lambda p: p.name)
    sources = []
    for f in files:
        name = f.name.replace("_preds.csv", "")
        df = pd.read_csv(f)
        if "sample_id" not in df.columns:
            df["sample_id"] = df.index + 1
        df["pred_label"] = df["pred_label"].astype(int)
        sources.append((name, df))
    return sources


def load_label_map() -> pd.DataFrame:

    df = pd.read_csv(LABELS_CSV)
    df.insert(0, "sample_id", df.index + 1)
    df["label"] = df["label"].astype(int)
    df = df.rename(columns={"label": "true_label", "filename": "filename"})
    if "item_id" not in df.columns:
        df["item_id"] = ""
    df["item_id"] = df["item_id"].fillna("")
    return df


def main() -> None:
    sources = load_pred_sources()
    labels = load_label_map()

    pred_by_id = {}
    for name, df in sources:
        for _, row in df.iterrows():
            sid = int(row["sample_id"])
            pred_by_id.setdefault(sid, {})[name] = (
                float(row["test_probs"]),
                int(row["pred_label"]),
            )

    source_names = [n for n, _ in sources]
    n_created = 0
    total_wrong = 0

    for name, _ in sources:
        folder = OUTPUT_ROOT / name
        folder.mkdir(parents=True, exist_ok=True)

        wrong_ids = []
        for sid, info in pred_by_id.items():
            _, pred = info[name]
            true_label = int(labels.loc[labels["sample_id"] == sid, "true_label"].iloc[0])
            if pred != true_label:
                wrong_ids.append(sid)

        for sid in sorted(wrong_ids):
            rec = labels.loc[labels["sample_id"] == sid].iloc[0]
            stem = Path(str(rec["filename"])).stem
            out_csv = folder / f"{sid:03d}_{stem}.csv"

            row_out = {
                "sample_id": sid,
                "filename": rec["filename"],
                "true_label": int(rec["true_label"]),
                "item_id": rec["item_id"],
            }
            wrong_in = []
            for src in source_names:
                pprob, ppred = pred_by_id[sid][src]
                result = "correct" if ppred == int(rec["true_label"]) else "wrong"
                row_out[f"{src}_prob"] = round(pprob, 6)
                row_out[f"{src}_pred"] = int(ppred)
                row_out[f"{src}_result"] = result
                if result == "wrong":
                    wrong_in.append(src)
            row_out["wrong_in"] = ";".join(wrong_in)
            row_out["n_sources_wrong"] = len(wrong_in)

            pd.DataFrame([row_out]).to_csv(
                out_csv, index=False, encoding="utf-8-sig"
            )
            n_created += 1

        total_wrong += len(wrong_ids)
        print(f"[{name}] 错检样本 {len(wrong_ids)} 个 -> {folder}")

    print(f"\n共生成 {n_created} 个错检样本 CSV (错检样本去重后 {total_wrong} 次判定)")


if __name__ == "__main__":
    main()
