# 用图像和文本证据复排

检索分两步。第一步从全库生成词项候选。第二步在同一批候选上对比多种排序器。文本版比较查询与标题、摘要和标签，像素版另加原图及相似照片的证据。每种排序器的返回阈值均由校准集确定。评测继续以工作簿中名为“高权重预期fid”的列为参考，每张清单外返回计一次多找。[实测成绩和适用范围](https://github.com/icarus0309/picfoundskills/blob/main/docs/evaluation-multimodal-v20.md)

## 准备输入

1. 按[图库准备说明](lexical-ranker.md)生成 `gallery/inventory.json`、`gallery/weighted_docs.json` 及与所选排序器相配套的 `facts.json`。同一份索引可供后续查询使用。
2. 本地安装 `pip install -r scripts/requirements.txt onnxruntime tokenizers`。文本编码使用 BGE-small-zh-v1.5，像素编码使用 Chinese-CLIP ViT-B/16。发行模型使用下表中固定的 ONNX 版本，下载后把 ONNX 文件和 tokenizer 放到各自目录的顶层。像素模型同时需要 `preprocessor_config.json`。
3. 新查询使用 JSON 对象，例如 `{"Q1":"找雨天在桥上撑伞的人"}`。普通查询沿用文本规划配置。已有同一查询的 `plans.json`、`contrast.json` 时可直接复用。

| 编码器 | ONNX 版本 |
|---|---|
| 中文文本 | [Xenova/bge-small-zh-v1.5](https://huggingface.co/Xenova/bge-small-zh-v1.5/tree/75c43b069aac4d136ba6bc1122f995fedcfd2781) 的 `onnx/model_quantized.onnx` |
| 图文 | [Xenova/chinese-clip-vit-base-patch16](https://huggingface.co/Xenova/chinese-clip-vit-base-patch16/tree/f26904860903e70e050b8f48255e5f48401816e9) 的 `onnx/model_quantized.onnx` |

以下命令从 Skill 目录运行。示例变量可替换成实际目录。

```powershell
$Data = 'D:/data/photos'
$Run = 'D:/runs/photo-search'
$Bge = 'D:/models/bge-small-zh'
$Clip = 'D:/models/chinese-clip'
python scripts/semantic_index.py --gallery "$Run/gallery" --model $Bge --out "$Run/gallery/text-index"
python scripts/clip_index.py --root $Data --inventory "$Run/gallery/inventory.json" --model-dir $Clip --out "$Run/gallery/clip-index"
```

文本索引对标题、摘要和去重后的标签编码。像素索引逐张编码图片并登记解码失败。使用含像素特征的排序器前需完成整库索引。图像最近邻可以提供同场景或近似照片的附加证据。排序器同时接收原有的词项、视觉事实、拍摄时间和地点特征。

## 使用随包模型

```powershell
python scripts/lexical_search.py --gallery "$Run/gallery" --queries "$Run/queries.json" --key-file "$Data/.secrets/deepseek.key" --model assets/semantic-ranker --text-model $Bge --clip-model $Clip --out "$Run/query-results"
```

已准备编码器后，可在同一条命令完成检索与校准后的单图复核。使用 `--review-config` 时会将边界图片及相应查询发送到配置的视觉端点。命令执行完会打印最终 CSV 路径。

```powershell
python scripts/lexical_search.py --gallery "$Run/gallery" --queries "$Run/queries.json" --key-file "$Data/.secrets/deepseek.key" --text-model $Bge --clip-model $Clip --photo-root $Data --review-config "$Data/.secrets/qwen.json" --out "$Run/reviewed-search"
```

同时提供文本和像素编码器时，入口自动使用随包校准选定的可迁移模型。仅传 `--text-model` 时使用文本语义模型，未传编码器时使用轻量词项模型。可显式指定 `--model`。本次随包图文模型默认启用具名场馆证据核验，最终表为 `venue-result/matches.csv`，`result/matches.csv` 保留未过滤的原始排序。加 `--no-require-venue-evidence` 可关闭场馆规则。单图复核遵循模型旁的 `review_recipe.json`，记录主排序器、提议审图的排序器及校准阈值。若配置两个排序器，二者重用同一批查询计划、候选及文本特征。

模型是否用到像素以 `model.json` 中的特征为准。仅使用文本语义的模型可以跳过像素索引步骤。输出目录中的 `result/matches.csv` 列出用例、图片路径、排序分及可见的词项证据。`result/seal.json` 记录本轮预测的哈希。

## 自行训练和复评

先按[原有训练说明](lexical-ranker.md)生成同一批查询的 `plans.json`、`contrast.json`、`split.json` 和 `visual-album` 特征目录。按顺序增加文本和像素特征。

```powershell
python scripts/semantic_features.py text --source "$Run/visual-album" --gallery "$Run/gallery" --plans "$Run/plans.json" --index "$Run/gallery/text-index" --model $Bge --out "$Run/with-text"
python scripts/semantic_features.py pixel --source "$Run/with-text" --gallery "$Run/gallery" --plans "$Run/plans.json" --index "$Run/gallery/clip-index" --model $Clip --out "$Run/with-pixels"
$Cases = 'D:/data/labels-v20.xlsx'
python scripts/lexical_ranker.py fit --root $Data --reference-workbook $Cases --gallery "$Run/gallery" --plans "$Run/plans.json" --split "$Run/split.json" --features "$Run/with-pixels" --out "$Run/fit"
python scripts/lexical_ranker.py predict --gallery "$Run/gallery" --plans "$Run/plans.json" --split "$Run/split.json" --features "$Run/with-pixels" --model "$Run/fit" --contrast "$Run/contrast.json" --out "$Run/sealed"
python scripts/lexical_ranker.py score --root $Data --reference-workbook $Cases --gallery "$Run/gallery" --out "$Run/sealed"
```

训练的 `--mine-semantic` 选项从词项、摘要、标签及像素多路得分中选取靠前的未列样本，和明确低权重、非预期及随机未列样本一并训练。未列样本仍按原协议降权。用 `--reference-workbook` 可直接指定修订工作簿。推理只需图库、查询和模型。

对于已有独立人工标注的固定图库，可试验 `semantic_feedback.py bank` 和 `augment`。该选项只从训练查询构建本地反馈库，排除与当前查询共用规范化核心词的记录。预测要求使用拟合时绑定的同一份反馈库。若只希望学习不同查询对应的视觉描述原型，可以用 `semantic_projection.py fit` 学习投影矩阵。训练特征使用 `augment --cross-fit`，线上使用完整矩阵。五折分组把相连的查询别名放在同一折。需要随模型保留 `projection.json` 与 `projection.npz`，新查询入口用 `--projection` 指定目录。


## 在选择边界逐张复核

对排序器刚好选中及刚好落选的照片分别取最多 `--depth` 张。Qwen 按主体、构图、属性与明确时间地点条件打 1–4 分。解码失败、拦截和无有效结论均保留原决定。深度和增删门槛需要在与业务相符的校准集确定。以下是该次文本模型验证使用的参数。默认排序命令不产生这一步的视觉服务费用。

```powershell
$Search = "$Run/text-search"
$Fit = 'assets/text-ranker'
python scripts/lexical_search.py --gallery "$Run/gallery" --queries "$Run/queries.json" --key-file "$Data/.secrets/deepseek.key" --model $Fit --text-model $Bge --out $Search
$Features = "$Search/text-features"
python scripts/boundary_review.py review --root $Data --gallery "$Run/gallery" --features $Features --model $Fit --source "$Search/result" --config "$Data/.secrets/qwen.json" --out "$Search/reviews" --workers 6 --depth 12
python scripts/boundary_review.py apply --gallery "$Run/gallery" --source "$Search/result" --review-dir "$Search/reviews" --out "$Search/reviewed" --depth 12 --drop 0 --add 3
```

若已有同库服务商拦截登记，在第一条命令增加 `--block-registry <已登记文件>`。复核从 `图片/` 读取原图并核对文件身份。支持中断续跑，各阶段分别封存。最终路径和直接审图分数见 `reviewed/matches.csv`。`--drop 0 --add 3` 表示原有结果全部保留，达到 3 或 4 分才补回原先未选中的图。多找仍按 F 列逐张扣分。

同一批已审照片可用于试验相似图片反馈。例如用 `scripts/example_rerank.py --help` 查看入口，传入原始排序结果、已封存的复核目录和文本索引。它比较其余候选与已审正反例的标题及摘要。参数 `--strength 0` 只使用直接复核结果，较大值会影响更多未审过的照片。输出中区分实际视觉评分与相似图片推算的支持分。

仓库根目录的 `tools/compare_semantic.py` 可以从已准备的查询、图库、索引和修订工作簿依次运行六种排序器，冻结校准选择，封存全部预测，再计分。运行 `python tools/compare_semantic.py --help` 查看输入。`tools/verify_semantic_report.py` 可在没有图库、工作簿和密钥时从公开逐条计数重算报告。

## 全库提名和等量重排对照

`scripts/dense_candidates.py` 可以从全库提名原候选之外的照片。输入为查询 JSON、已准备图库、文本与像素索引及对应编码器。每条查询分别从标题、摘要和原图取 `--top` 张，输出带渠道的 `proposals.json` 和校验哈希。与既有候选取并集后仍需精排。报告按每路 50 张实测计入的正确和额外关系。

`scripts/cross_rerank.py` 将查询、图片摘要和地点等字段联合送入 mMARCO 交叉编码器，维持每条查询的原返回数。将以下版本中的 `tokenizer.json` 与 `onnx/model_quint8_avx2.onnx` 放在同一个模型目录，调用时设置 `--encoder <目录> --weight 0.25`，另提供相同的 `--gallery --features --model --source --out`。输出的 score 为名次倒数，evidence 保留原排序得分和交叉编码器分数。校准和主测试成绩见报告。[固定的 mMARCO 模型版本](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/1427fd652930e4ba29e8149678df786c240d8825)

逐图复核接受整数及仅含一个 1–4 数字的字符串。用 `scripts/replay_review.py --gallery <图库> --source <原始排序结果> --review-dir <原视觉复核> --out <新目录>` 可从缓存重放。新运行封存原始响应文件摘要，重放前核对。较早运行可核对请求身份、图片哈希和已知票，并在重放时封存现有缓存。若旧配置的端点带有末尾斜线，使用 `--legacy-endpoint-url` 提供当时的确切 URL。

`scripts/transfer_review.py` 可将一套排序器收集的逐图评分应用到另一套排序器。只有图库、查询编号及文本完全相同，且图片在目标排序器候选中时才可转用。参数 `--scout` 和 `--target` 分别指定两套原始封存结果，`--reviews` 指定前者的复核目录。另指定 `--gallery --out --depth --drop --add`。新增项附审图分数及来源排序分数，保留原选项的目标排序分数。
