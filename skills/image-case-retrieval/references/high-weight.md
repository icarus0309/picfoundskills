# 高权重图片检索与评分

当目标是找齐高权重照片且处罚多返回的照片时，采用本模式。每个用例对每张图单独计数。精确率为高权重命中数除以实际返回数，覆盖率为高权重命中数除以该列所有去重的 fid 数。低权重、明确非预期、未列入清单的返回图均计一次多找。不同用例可以重复计同一张图。整行高权重清单为空时单列该用例，汇总分数只使用有参考值的用例。

## 执行

沿用技能首页列出的目录与凭据布局。以下命令在技能目录中运行。

```powershell
python -m pip install -r scripts/requirements.txt
python scripts/weighted_retrieval.py --root <数据根目录> --out <新运行目录> --offset 0 --limit 10 --vlm-config <Qwen配置>
python scripts/score_weighted.py --root <数据根目录> --out <新运行目录>
```

若已有同一图库、同一查询的盲测视觉结果，可在第一条推理命令追加 `--visual-prior <视觉批次>/predictions.json`，将视觉命中和待核实图加入候选。给定 `--block-registry <历史运行>/provider_blocks.json` 可继承同一图库的服务商图像拦截登记。支持 `--cache-root <旧运行目录>` 复用按模型、端点、输入与图片摘要匹配的成功调用。

按表头名称读取“高权重预期fid”，适用于 F 列和列位置发生变化的工作簿。表头文字不同时显式传入 `--label-header`。若要重评旧版的已封存预测，使用下列命令。

```powershell
python scripts/score_weighted.py --legacy --root <数据根目录> --out <含有predictions.json及封存的批次目录>
```

## 检索与复核

1. 读取用例 ID、原始查询以及图库索引。词表扩展、全库描述筛选、已有视觉结果提名分别构建候选，目录不承担排除作用。
2. 文本模型分开阅读标题、摘要、长描述、分类词、索引词、日期和地点，对候选评定 0 至 4 级。语义描述缺失的图片也进入单图复核。文本等级 2、3 的边界结果交给 Qwen 查看像素。等级 4 可以由描述直接确认，证据来源应在诊断时保留。服务商拦截、无有效结论和解码失败保留为空值。
3. 描述与视觉均有判断时取均值，基础模式的返回阈值为 3。查看 `weighted_matches.csv` 获取本机图片相对路径，`weighted_predictions.json` 包含分数，`weighted_scores.json` 包含每条用例和全体的命中、多找、漏找。候选覆盖与最终返回覆盖分别计算。以上材料可能含私有路径和用例信息。
4. 需要定位漏图时，按顺序查找 `weighted_candidates.json`、`weighted_expanded.json`、`weighted_grades.json`、`weighted_visual/` 和最终预测。检查项目依次为初筛、扩展召回、描述分级、单图复核和最终阈值。候选池允许较多弱匹配。发布结果时以最终返回集合计算精确率和覆盖率。

## 可选的排序对照

`weighted_rerank.py --root <数据根目录> --source <高权重运行目录> --out <新目录>` 会对同一查询的候选并列评分，再与已取得的视觉判断融合。强度阈值为 7。使用 `score_weighted.py` 对同一组 fid 独立评分。

本地像素对照另需 `numpy`、`onnxruntime`、`tokenizers`。将模型目录中的 `onnx/model_quantized.onnx` 保存为 `model_quantized.onnx`，并在同一目录保留 `preprocessor_config.json`、`tokenizer.json`。试验使用固定的 [Chinese-CLIP ONNX 版本](https://huggingface.co/Xenova/chinese-clip-vit-base-patch16/tree/f26904860903e70e050b8f48255e5f48401816e9)。

```powershell
python scripts/clip_index.py --root <数据根目录> --inventory <高权重运行目录>/inventory.json --model-dir <模型目录> --out <向量目录> --prioritize <高权重运行目录>/weighted_grades.json --only-priority
python scripts/clip_fusion.py --source <高权重运行目录> --index <向量目录> --model-dir <模型目录> --out <融合目录>
python scripts/score_weighted.py --root <数据根目录> --out <融合目录>
```

`--only-priority` 对指定文件中至少获得一级描述分的图像建索引。去掉它可索引整库。融合阶段使用原始查询与照片像素的相似度、相邻图片的相似度，以及高置信检索结果提供的正反例。封存目录保存模型摘要、实际向量数和结果阈值。每种方法使用独立输出目录。在开发集确定阈值后固定到验证集，两项指标以同一批返回图计算。
