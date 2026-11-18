# DL-Homework

深度学习课程设计：基于 YOLO26 的目标检测。两个数据集——**启辰（qichen）** 试卷批改符号检测（handwriting / tick / cross 三类）与 **勤思（qinsi）** 人形检测（person 单类）。

评测指标为两个测试集 mAP50-95 的平均值。

## 目录结构

```
src/                 全部训练、推理、评估脚本
  train_exp.py       ultralytics 训练包装（补齐官方脚本未暴露的增强参数）
  make_split.py      去泄漏 + 类别分层的训练/验证划分
  oversample.py      稀有类过采样
  enhance.py         勾叉 copy-paste 合成（乘性混合贴图）
  predict_tta.py     TTA 推理，复用官方 schema
  predict_ensemble.py  多模型 WBF 融合推理
  predict_submit.py  一键产出提交包
  eval_ensemble.py   COCO 口径的融合评估
  eval_sweep.py      置信度阈值 / TTA 扫描
  eval_split.py      在训练集上评估，用于测泛化 gap
  diagnose_iou.py    逐类 IoU 分布诊断（定位瓶颈）
  box_scale.py       框缩放扫描（检验系统性框偏置）
  wbf_lib.py         推理 + 加权框融合底层实现
  run_*.sh, chain*.py  实验队列（作为实验过程记录保留）

experiments/<run>/   每组的 args.yaml、config.json、results.csv、训练曲线、混淆矩阵
experiments/summary.md  全部实验汇总表
weights/qichen/      启辰 6 个模型权重（参与最终融合）
weights/qinsi/       勤思 2 个模型权重
data/                数据划分配置与类别文件
docs/                实验设计与分析文档
submissions/         提交包
```

### 关于两个大权重

`weights/qichen/q_combo_m-best.pt` 与 `weights/qinsi/n_m_aug-best.pt` 各 44 MB，超过 GitHub
Git Data API 单次上传上限，因此放在 Release 资产里，下载后放回上表对应路径即可：

```bash
curl -L -o weights/qichen/q_combo_m-best.pt \
  https://github.com/ZJR-FZD/DL-Homework/releases/download/weights-v1/q_combo_m-best.pt
curl -L -o weights/qinsi/n_m_aug-best.pt \
  https://github.com/ZJR-FZD/DL-Homework/releases/download/weights-v1/n_m_aug-best.pt
```

## 环境

- Python 3.11、PyTorch 2.4.0+cu121、ultralytics 8.4.215（支持 YOLO26）
- 单卡 RTX 4090 24G
- 复现依赖：`pip install ultralytics pycocotools ensemble-boxes pydantic pydantic-settings`

## 数据划分

课程提供的数据集只有 `train` 有标注，`val` 用于隐藏评测。因此自建验证集：

1. **去泄漏**：启辰训练集里有 51 对完全重复的图片（MD5 相同），按重复组整体切分，避免同一张图同时出现在训练和验证集
2. **分层**：按"是否含 tick / 是否含 cross"分桶，各类按比例抽取
3. 结果：启辰 1610 / 180，勤思 5709 / 634

```bash
python src/make_split.py --root . --out split
```

## 实验结果

### 启辰（qichen）

| 实验 | 模型 | imgsz | mAP50-95 | mAP50 | 说明 |
|---|---|---|---|---|---|
| q_s1024_s | yolo26s | 1024 | 0.3140 | 0.5847 | 基线，默认增强 |
| q_hsv | yolo26s | 1024 | **0.3301** | 0.6050 | 弱化色彩抖动，保住"红墨 vs 黑字"线索 |
| q_mos05 | yolo26s | 1024 | 0.3295 | 0.6236 | mosaic 0.5，避免拼图缩小大勾 |
| q_s1280_s | yolo26s | 1280 | 0.3243 | 0.6081 | 更高分辨率 |
| q_combo | yolo26s | 1024 | 0.3227 | 0.5880 | 过采样（cross×4）+ 各项先验 |
| q_combo_s1 | yolo26s | 1024 | 0.3232 | 0.6041 | 同上，换随机种子 |
| q_combo_m | yolo26m | 1024 | 0.3345 | **0.6402** | 换骨干 |
| q_mos0 | yolo26s | 1024 | 0.3120 | 0.5891 | 完全关 mosaic，反而更差 |
| q_long | yolo26s | 1024 | **0.3357** | 0.6034 | 跑满 120 轮不早停，单模型最佳 |

### 勤思（qinsi）

| 实验 | 模型 | imgsz | mAP50-95 | mAP50 |
|---|---|---|---|---|
| n_s640_s | yolo26s | 640 | **0.7258** | 0.9831 |
| n_m_aug | yolo26m | 640 | 0.7212 | 0.9824 |
| n_s960 | yolo26s | 960 | 未完成 | 训练中途终止（`experiments/n_s960/` 仅作记录） |

### 多模型融合（统一 COCO 口径）

| 组合 | 启辰 | 说明 |
|---|---|---|
| 单模型 | 0.3148 | — |
| 2 模型 | 0.3270 | q_hsv + q_mos05 |
| 4 模型 | 0.3503 | + q_s1280_s + q_s1024_s |
| **5 模型** | **0.3591** | + q_combo_m，**最终采用** |
| 6 / 7 模型 | 0.3577 / 0.3574 | 加入质量较差的过采样版后回落 |

| 组合 | 勤思 | 说明 |
|---|---|---|
| 单模型 | 0.7194 | n_s640_s |
| **2 模型** | **0.7283** | + n_m_aug，**最终采用** |

## 关键分析

### 1. 提交阈值会白丢 3 个点

评测把 result.json 里的框按置信度排序扫 PR 曲线，但文件里只写入超过阈值的框，等于曲线被截断：

| 提交阈值 | mAP50-95 |
|---|---|
| 0.001 | 0.3141 |
| 0.01 | 0.3107 |
| 0.25（官方脚本默认） | 0.2816 |

### 2. TTA 增益

`conf=0.01` 下 0.3107 → 0.3261（+1.5），cross 从 0.145 升到 0.185。官方推理脚本不支持 `augment`，因此重写了 `predict_tta.py` 并复用官方 Pydantic schema。

### 3. 瓶颈是召回和误检，不是框精度

逐类统计预测框与真值框的 IoU（`diagnose_iou.py`）：

| 类别 | 匹配后 IoU 中位 | IoU>0.5 占比 | 召回@0.3 | 误检 |
|---|---|---|---|---|
| handwriting | 0.819 | 96% | 0.930 | 372 |
| tick | 0.796 | 96% | 0.667 | 115 |
| cross | 0.697 | 93% | 0.636 | 19 |

框只要命中，IoU 就在 0.8 上下，定位不是主要瓶颈。

### 4. 启辰已接近标注可达上限

用 `eval_split.py` 在**训练集**上评估（模型见过的图）：

| 类别 | 训练集 mAP50-95 | 验证集 | 差距 |
|---|---|---|---|
| handwriting | 0.600 | 0.550 | 0.050 |
| tick | 0.393 | 0.279 | 0.114 |
| cross | 0.145 | 0.175 | **-0.030** |
| 全部 | 0.379 | 0.330 | 0.049 |

- 总体泛化差距仅 0.049，说明不是过拟合，而是标注本身的框规范一致性限制了上限
- **cross 在训练集上反而低于验证集**，是标注口径前后不一致的直接证据
- 融合后 0.3591，已达训练集上限的 95%

### 5. 勤思的差距是随机抖动

训练集 0.870 → 验证集 0.726，差距 0.144。三个假设被逐个排除：

| 检查 | 结果 |
|---|---|
| 推理分辨率（512~768） | 0.718~0.725 全平，mAP50 恒 0.983 |
| 训练/验证数据分布 | 图片尺寸构成、框相对面积中位数一致（0.02119 vs 0.02128） |
| 框系统性偏置（0.94~1.04 缩放） | 最优恰为 1.00 且左右对称，无偏置 |

结论：误差是零均值的随机抖动，只能靠多模型融合平均掉——实测 2 模型融合 +0.9 点，与这个判断一致。

### 6. 负结果

- **过采样无效**：cross 只有 108 张图，重复 4 倍后模型记住了固定外观而非泛化（0.3227 vs 0.3301）
- **重增强无效**：勤思加 mixup / 强色彩抖动 / yolo26m 后 0.7212，低于简单配置的 0.7258
- **mosaic 不能全关**：0.3120，比 mosaic 0.5 低 1.8 点
- **copy-paste 合成有技术障碍**：alpha 抠图会在目标图留下灰色矩形框（源纸比目标纸暗），改用乘性混合解决；但训练集裁出的勾叉有近半混入印刷表格和文字，需连通域筛选，未达可用质量

## 复现命令

```bash
# 1. 划分数据
python src/make_split.py --root . --out split

# 2. 训练（示例：启辰最佳单模型配置）
python src/train_exp.py --dataset qichen --model yolo26s.pt --imgsz 1024 \
  --batch 12 --epochs 120 --patience 100 --hsv-h 0.005 --hsv-s 0.25 \
  --hsv-v 0.30 --fliplr 0.0 --mosaic 0.5 --close-mosaic 20 --name q_long

# 3. 融合评估
python src/eval_ensemble.py --dataset qichen \
  --members "weights/qichen/q_hsv-best.pt:1024,weights/qichen/q_mos05-best.pt:1024"

# 4. 生成提交包
python src/predict_submit.py \
  --qichen "weights/qichen/q_hsv-best.pt:1024,weights/qichen/q_mos05-best.pt:1024" \
  --qinsi "weights/qinsi/n_s640_s-best.pt:640" --tag v3 --group-id 0
```

## 提交包

`submissions/` 下的 zip 结构为：

```
qichen_val_labels/result.json
qinsi_val_labels/result.json
```

最终版本为 5 模型融合 + TTA + 阈值 0.01。
