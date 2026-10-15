# YOLO11 分类模式对比实验 (`YOLO11/`)

用 **YOLO11 分类任务** (`yolo11n-cls.pt` 等) 直接对整张试纸条 RGB 图像做阴/阳性二分类,
作为与项目双流分类方法 (`DualStreamStripNet` / C / D / E) 的对照基线。
输入格式与图像流完全一致, 无需额外标注。

所有输出统一落在 **`outputs/YOLO11/`**。

---

## 0. 准备数据集 (必读, 需手动完成)

数据集**不再由脚本从项目标注自动生成**。请自行准备三个划分目录 (ImageFolder 结构),
然后在 **`YOLO11/dataset_config.py`** 顶部的配置区填写路径 (已留白, 等你去填):

```python
TRAIN_DIR = r""      # 训练集目录 (目录下含 negative/ 与 positive/)
VAL_DIR   = r""      # 验证集目录
TEST_DIR  = r""      # 测试集目录
```

期望目录结构 (类别文件夹名大小写不敏感, 支持 `negative`/`positive` 及 `N`/`P`、`0`/`1` 等别名):

```
<你的数据根>/
├─ train/
│  ├─ negative/       # 阴性 (标签 0)
│  └─ positive/       # 阳性 (标签 1)
├─ val/
│  ├─ negative/
│  └─ positive/
└─ test/
   ├─ negative/
   └─ positive/
```

- 路径可写绝对路径, 也可写相对项目根目录的路径。
- 三个划分若在同一父目录 (如上), 训练时直接把该父目录交给 Ultralytics (**零拷贝**);
  否则会自动归一化复制到 `outputs/YOLO11/dataset` (类别目录统一为 `N`/`P`) 后使用。
- 样本数与默认期望值 (386/50/80) 不一致时**仅告警不中断** (见 `ALLOW_COUNT_MISMATCH`)。
- 未填写路径时, 运行会给出清晰提示并停止。

---

## 1. 与主项目保持一致的约定

| 项目 | 取值 |
| --- | --- |
| 数据集 | 由 `YOLO11/dataset_config.py` 指定 (默认期望 训练 386 / 验证 50 / 测试 80) |
| 标签定义 | `0 = negative (阴性)` / `1 = positive (阳性)` (ImageFolder 字母序 `negative → 0`, `positive → 1`) |
| 输入尺寸 | `--imgsz 224` (默认) 或 `--imgsz 320`, 整图 resize + CenterCrop |
| 随机种子 | `42, 831, 2026, 7, 2691` (5 个, 与论文设定一致) |
| 决策阈值 | 仅在 **验证集** 上用约登指数 (Youden J) 选取; 测试集只用于最终报告 (无泄漏) |
| 指标口径 | 复用 `src/evaluation/metrics.compute_metrics`; AP 为项目统一口径 (201 点 recall 网格右侧最大 precision 包络的梯形积分); AUC 95% CI 用 bootstrap (`compute_auc_ci`) |

---

## 2. 环境

已确认可用环境: `D:\Anaconda_3\envs\TestStrips_py3.9_cu128` (torch 2.8.0+cu128 + ultralytics 8.4.148)。
若该环境缺失依赖:

```powershell
& "D:\Anaconda_3\envs\TestStrips_py3.9_cu128\python.exe" -m pip install ultralytics
```

`base` 环境与 `TestStrips_py3.9` 环境 **没有** ultralytics, 请显式使用上面的解释器。

---

## 3. 一键运行 (推荐)

在项目根目录 `d:\work\Intelligent-Recognition-of-Biological-Test-Strips` 下:

```powershell
# 方式 A: PowerShell 包装脚本 (自带正确解释器, 默认 5 个种子 / 224 输入 / 100 epochs)
powershell -ExecutionPolicy Bypass -File YOLO11\run_all.ps1

# 320x320 输入
powershell -ExecutionPolicy Bypass -File YOLO11\run_all.ps1 -Imgsz 320

# 只重新评估 (复用已训练权重)
powershell -ExecutionPolicy Bypass -File YOLO11\run_all.ps1 -SkipTrain

# 方式 B: 直接用解释器调用统一步骤入口
& "D:\Anaconda_3\envs\TestStrips_py3.9_cu128\python.exe" YOLO11\run_all.py
& "D:\Anaconda_3\envs\TestStrips_py3.9_cu128\python.exe" YOLO11\run_all.py --imgsz 320 --epochs 150
& "D:\Anaconda_3\envs\TestStrips_py3.9_cu128\python.exe" YOLO11\run_all.py --model yolo11s-cls.pt --seeds 42 831
```

`run_all.py` 依次执行: **校验数据集 → 5 个种子训练 → 评估/集成/出图**。

---

## 4. 分步运行

```powershell
$py = "D:\Anaconda_3\envs\TestStrips_py3.9_cu128\python.exe"

# 步骤 1: 校验 YOLO11/dataset_config.py 配置的数据集
& $py YOLO11\prepare_dataset.py                 # 校验 + 必要时归一化复制
& $py YOLO11\prepare_dataset.py --verify-only    # 只校验, 不复制
& $py YOLO11\prepare_dataset.py --force          # 归一化复制时强制覆盖

# 步骤 2: 多随机种子训练 (默认 42 831 2026 7 2691)
& $py YOLO11\train_yolo11_cls.py --imgsz 224 --epochs 100
& $py YOLO11\train_yolo11_cls.py --seeds 42 831 --imgsz 320
& $py YOLO11\train_yolo11_cls.py --model yolo11s-cls.pt --epochs 150

# 步骤 3: 验证/测试评估 + 5 种子 soft-voting 集成 + 出图
& $py YOLO11\test_yolo11_cls.py                                   # 自动选择最近训练的 tag
& $py YOLO11\test_yolo11_cls.py --tag yolo11n-cls_224
& $py YOLO11\test_yolo11_cls.py --tag yolo11n-cls_320 --imgsz 320
& $py YOLO11\test_yolo11_cls.py --no-figures                      # 不出图
& $py YOLO11\test_yolo11_cls_svg.py                            # 仅用 reports/ 数据重建 SVG (无需权重)
```

> ⚠ `test_yolo11_cls.py --imgsz` 必须与训练时的 `--imgsz` 一致 (预处理尺寸不同会显著影响结果)。

---

## 5. 输出目录结构

```
outputs/YOLO11/
├─ dataset/                       # 归一化副本 (仅当三个划分不在同一父目录时生成)
│  ├─ train/{N,P}/
│  ├─ val/{N,P}/
│  └─ test/{N,P}/
├─ weights/yolo11n-cls.pt         # 预训练权重缓存 (避免污染项目根目录)
├─ runs/<tag>/seed_<seed>/        # tag 形如 yolo11n-cls_224
│  ├─ weights/best.pt, last.pt
│  ├─ results.csv                 # 逐 epoch 损失/准确率
│  └─ results.png, confusion_matrix.png, ...
├─ reports/
│  ├─ dataset_summary.csv                 # 数据集构建校验结果
│  ├─ train_config_<tag>.json             # 训练超参与环境指纹
│  ├─ train_runs_<tag>.json               # 各种子 best.pt 路径
│  ├─ logs/train_<tag>_seed<seed>.log     # 每个种子的完整训练日志
│  ├─ per_seed_metrics.csv                # 每种子 × 划分 × 阈值来源 的指标
│  ├─ per_seed_predictions_{val,test}.csv # 逐样本概率 (含 ensemble 行)
│  ├─ ensemble_metrics.csv                # 集成指标
│  ├─ ensemble_predictions_{val,test}.csv # 集成逐样本概率
│  ├─ ensemble_summary.csv                # ★ 论文可直接引用的汇总表
│  ├─ leaky_reference_test_youden.csv     # 仅参考: 测试集直接寻优阈值 (有泄漏)
│  ├─ eval_config_<tag>.json
│  └─ gradcam_<图片名>.json               # 单图 Grad-CAM 的配置与各成员概率
├─ heatmaps/                              # 单图 Grad-CAM 三联图 (逐种子 + 等权平均集成)
│  └─ cam_yolo11_<tag>_<seed>_<图片名>_true_<标签>_pred_<POS|NEG>_<概率>.png
└─ figures/<tag>/
   ├─ roc_test.svg / roc_val.svg
   ├─ pr_test.svg
   ├─ confusion_matrix_test.svg
   └─ threshold_scan_test.svg
```

---

## 6. 结果解读

`reports/ensemble_summary.csv` 含三行, 分别对应:

1. **YOLO11-cls ensemble (soft voting)** — 5 个种子概率平均, 阈值 = 集成验证集概率的约登阈值 → 主结果;
2. **best single (by val AUC)** — 只依据验证集 AUC 挑种子, 无泄漏 → 可作对照;
3. **best single (by test AUC)** — 用测试集挑种子, **存在选择性偏差, 论文不建议作为主结果**。

`per_seed_metrics.csv` 中每行含 `auc / auc_ci_low / auc_ci_high / ap / accuracy /
sensitivity / specificity / precision / f1_score / tp / fp / tn / fn`,
`threshold_source` 区分 `fixed_0.5` 与 `val_youden` (论文主指标应使用后者)。

与主项目方法对比时, 请使用同一测试集 (80 张) 与同一阈值策略 (验证集约登) 下的
`test_auc` / `test_ap` / `test_sensitivity` / `test_specificity`。
主项目各方法 (Tweak/C/D/E) 的集成报告见
`outputs/reports/new_env_test/ensemble_report/` 与 `outputs/checkpoints/<模式>/reports/`。

---

## 7. 单图 Grad-CAM 热力图

借鉴项目根目录 `GradCAM.py`, 提供 YOLO11 分类模型的单图可解释性输出
(脚本: `gradcam_yolo11.py`), 与评估脚本共用同一套权重发现与阈值口径。

```powershell
$py = "D:\Anaconda_3\envs\TestStrips_py3.9_cu128\python.exe"

# 默认: 自动选最近训练的 tag, 用其全部已训练种子逐个出图 + 等权平均集成图 (6 张)
& $py YOLO11\gradcam_yolo11.py --image data/test/raw/N_048_1.png

# 单种子 (最接近 GradCAM.py 的单模型用法)
& $py YOLO11\gradcam_yolo11.py --image data/test/raw/N_048_1.png --seed 42

# 指定 tag / 尺寸 / 固定解释"阳性"类别 / 换目标层 (9=C2PSA, 8=C3k2)
& $py YOLO11\gradcam_yolo11.py --image data/test/raw/N_003_1.jpg `
    --tag yolo11n-cls_224 --imgsz 224 --class-index 1 --target-layer 9

# 直接指定权重 (不依赖 runs 目录)
& $py YOLO11\gradcam_yolo11.py --image data/test/raw/N_003_1.jpg `
    --weights outputs\YOLO11\runs\yolo11n-cls_224\seed_42\weights\best.pt
```

要点:

- **目标层**: 默认取分类头 `Classify` 内部的 1×1 卷积 (`net.model[-1].conv`), 即最后一个
  输出 4D 特征图的卷积层 (YOLO11-cls: 0~8 主干 → 9=C2PSA → 10=Classify), 输出 7×7×1280;
  可用 `--target-layer 9` 改看 C2PSA 特征 (7×7×256)。
- **预处理**: 复用 `load_val_transform`, 与训练验证/测试评估完全一致; 显示用原图用同一套
  Resize+CenterCrop 重建, 三联图空间严格对齐。
- **被解释类别**: `--class-index auto` (默认) = 该模型预测类别; 也可固定 0/1。
- **阈值**: `--threshold` > `reports/eval_config_<tag>.json` 的 `val_thresholds[seed]`
  (单种子) 或 `ensemble_threshold` (集成) > 0.5。
- **多种子**: 每个种子单独出一张图, 并额外输出 `ensemble<N>` 等权平均 CAM (概率同为等权平均,
  与 `test_yolo11_cls.py` 的 soft-voting 口径一致)。
- **输出**: `outputs/YOLO11/heatmaps/cam_yolo11_<tag>_<seed>_<图片名>_true_<标签>_pred_<POS|NEG>_<概率>.png`
  (「原图 | 热力图 | 叠加图」三联图, 300 dpi); 本次配置与各成员概率落盘
  `outputs/YOLO11/reports/gradcam_<图片名>.json`。

> 注: 由于热力图路径打开了梯度 (Ultralytics 默认 `requires_grad=False`, 脚本会显式打开),
> cuDNN 可能选用不同的卷积算法, 其阳性概率与评估 CSV 有约 1e-4 的数值差异, 不影响判定与可视化。

---

## 8. 脚本清单

| 文件 | 作用 |
| --- | --- |
| `dataset_config.py` | ★ 数据集路径配置 (手动填写 `TRAIN_DIR` / `VAL_DIR` / `TEST_DIR`) |
| `yolo11_common.py` | 公共模块: 路径/种子/类别约定、数据集定位与校验、与项目一致的评价口径、概率推理 |
| `prepare_dataset.py` | 校验配置的数据集并 (必要时) 归一化复制 |
| `train_yolo11_cls.py` | 指定/默认多种子训练 YOLO11 分类模型, 日志与配置落盘 |
| `test_yolo11_cls.py` | 验证/测试评估、多阈值口径指标、多种子集成、CSV 与图表输出 |
| `gradcam_yolo11.py` | 单图 Grad-CAM 热力图 (逐种子 + 多种子等权平均 CAM) |
| `run_all.py` | 一键串行执行数据集校验 → 训练 → 评估三步 |
| `run_all.ps1` | PowerShell 包装 (自动使用 cu128 解释器, 默认 5 种子) |
