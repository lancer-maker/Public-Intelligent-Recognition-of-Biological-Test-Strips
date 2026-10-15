
import os
import csv
from pathlib import Path


def parse_filename(filename: str):

    base, ext = os.path.splitext(filename)
    if ext.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
        return None
    parts = base.split('_')

    if parts[0].startswith('P') and len(parts) == 3:
        label = parts[2]
        if label in ('0', '1'):
            return label, parts[1]
    if parts[0] in ('N', 'V') and len(parts) == 3:
        label = parts[2]
        if label in ('0', '1'):
            return label, ''
    return None


def main():
    IMAGE_DIR = r"data\test\raw"
    OUTPUT_CSV = r"data\test\labels"

    rows = []
    for filepath in Path(IMAGE_DIR).glob('*'):
        if not filepath.is_file():
            continue
        result = parse_filename(filepath.name)
        if result is None:
            continue
        label, item_id = result
        rows.append([filepath.name, label, item_id])

    output_path = Path(OUTPUT_CSV)
    if output_path.suffix == "":
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / "labels.csv"
        else:
            output_path = output_path.with_suffix('.csv')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label', 'item_id'])
        writer.writerows(rows)

    print(f"生成完成，共 {len(rows)} 条记录 → {output_path}")


if __name__ == '__main__':
    main()