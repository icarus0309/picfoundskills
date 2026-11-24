# 全库倒排检索与高权重排序

适用于有图片描述、标签或 OCR 的图库，以及需要按高权重 fid 清单严格计分的任务。发布包带有训练好的数值模型。查询和图库可以更换。每个结果含相对路径、排序分和可检查的词与字段。排序分用于比较同次查询的候选，不是相关概率。

## 首次建立图库

在 Skill 目录运行以下 PowerShell 命令。图库根目录需要 `图片/` 和 `ES_byFileType_user10_20261117170358.xlsx`。建库与新查询推理都可以在没有历史用例工作簿的环境运行。

```powershell
python -m pip install -r scripts/requirements.txt
$Data = 'D:/data/photos'
$Run = 'D:/runs/photo-search'
python scripts/lexical_facts.py --root $Data --out "$Run/gallery"
```

若使用的 `model.json` 中 `visual_enrichment` 为 true，继续提取可见事实。首次训练使用下列主题词，另会选取缺标题、缺长描述及拼贴、九宫格、组图。词表为定向补充，首轮迁移评测宜保留词表。每张被选图片只需补充一次，后续新查询共享缓存。已有服务商拦截登记时，增加 `--block-registry <登记文件>`，登记文件同目录需有对应 `inventory.json`。

```powershell
python scripts/lexical_facts.py --root $Data --out "$Run/gallery" --vlm-config "$Data/.secrets/qwen.json" --focus 大门 入口 门楼 鸟居 牌坊 园门 动物园 水族馆 水族箱 鱼群
```

Qwen 的私有配置使用 `base_url`、`model`、`key_file`。缺证据、审核拦截和解码失败按状态保留。文本模式也能搜索有历史描述的图片。

## 检索新查询

将以下 JSON 以 UTF-8 存为 `queries.json`。键是自定义查询编号。

```json
{"Q1":"找钟面清楚可见的钟楼照片","Q2":"2019年在杭州拍的照片"}
```

```powershell
python scripts/lexical_search.py --queries ./queries.json --gallery "$Run/gallery" --out "$Run/query-01" --key-file "$Data/.secrets/deepseek.key"
```

打开 `query-01/result/matches.csv` 可按用例查看返回图片和命中证据。`result/predictions.json` 同时给出全部候选、字面搜索结果及最终结果，包括返回零张的查询。新任务使用新输出目录。同一目录可续跑。若手头已有与本次查询对应的 `plans.json` 和本模型所需的 `contrast.json`，传入 `--plans ... --contrast ...` 即可离线复现，不需提供 API 密钥。

需要明确地名证据时，可在同一检索命令加上 `--require-venue-evidence`。具名场所查询的每个地点组都需命中可检查的字段。原始表保存在 `result/matches.csv`，带地点证据的返回保存在 `venue-result/matches.csv`。37 条追加测试中，这个开关使精确率从 83.00% 提高到 85.63%，覆盖率从 51.52% 变为 51.16%。从仓库根运行 `python tools/evaluate_venue_filter.py --root <数据根目录> --run <主实验目录> --out <结果.json>` 可复测。

倒排索引在整个图库中查找中文词项和双字片段。模型使用标题、摘要、长描述的首句与背景、标签词及位置、OCR、主体与动作、时间地点等证据。用于对照的扩展版本也会考虑同城 30 分钟内照片的弱证据。目录版本只计算各相册中词项的支持比例，保留其全部输入候选。模型可以同时使用语义弱关联与干扰词降权。训练特征不含 fid、文件名、查询编号或目录编号。

## 重新训练和比较

历史用例表 A/B 是用例编号和文本，F/G/H 分别是高权重、低权重、非预期 fid。以下示例训练“语料词表＋视觉补充”版本。先冻结主题留出，再读取训练和校准标签。换图库重新训练时，按相同原则设置适合该评测的主题。对于未给高权重标签的用例，记录返回但单列为不可评分。

```powershell
python scripts/lexical_plans.py --root $Data --out "$Run/plans.json"
python scripts/lexical_split.py --plans "$Run/plans.json" --out "$Run/split.json"
python scripts/lexical_contrast.py --root $Data --gallery "$Run/gallery" --plans "$Run/plans.json" --out "$Run/contrast.json"
python scripts/lexical_features.py --gallery "$Run/gallery" --plans "$Run/plans.json" --contrast "$Run/contrast.json" --facts "$Run/gallery/facts.json" --out "$Run/enriched"
python scripts/lexical_ranker.py fit --root $Data --gallery "$Run/gallery" --features "$Run/enriched" --plans "$Run/plans.json" --split "$Run/split.json" --out "$Run/fit-enriched"
python scripts/lexical_ranker.py predict --gallery "$Run/gallery" --features "$Run/enriched" --plans "$Run/plans.json" --split "$Run/split.json" --model "$Run/fit-enriched" --contrast "$Run/contrast.json" --out "$Run/pred-enriched"
python scripts/lexical_ranker.py score --root $Data --gallery "$Run/gallery" --out "$Run/pred-enriched"
```

五组对照共用同一图库、主题划分、训练方法和校准网格。

| 运行名称 | 构造特征 |
|---|---|
| `base` | 调用 `lexical_features.py`，仅传图库、计划和输出目录 |
| `profile` | 在 base 的参数上增加 `--contrast` |
| `album` | 执行 `lexical_album.py --features "$Run/profile" --gallery "$Run/gallery" --out "$Run/album"` |
| `enriched` | 在 profile 的参数上增加 `--facts` |
| `visual-album` | 对 enriched 执行同一目录统计脚本，输出目录改为 visual-album |

对每组特征分别执行 `fit`，模型目录命名为 `fit-<运行名称>`。在仓库根目录运行 `python tools/lexical_benchmark.py freeze --run <运行根目录>`，以校准结果选择版本。对每组执行 `predict`，结果存入 `pred-<运行名称>`。全部封存后再逐组执行 `score`。已有私有运行可用 `tools/lexical_benchmark.py export --run <运行根目录> --out <匿名导出目录>` 汇总。单独验证公开数字使用 `python tools/lexical_benchmark.py verify`。需要审阅错误时，用 `tools/audit_lexical_boundaries.py --root <图库根目录> --run <运行根目录> --out <私有审阅目录>` 输出盲审表和独立答案表。

训练和阈值校准只使用各自的主题分区。未列项在学习中降权，在评分中按一次多找足额扣分。表内遗漏、同名歧义及不同图库的标签边界都会影响结果。请报告实测精确率、覆盖率和命中／多找／漏找的计数。
