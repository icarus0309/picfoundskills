# 安装与运行

## 环境与安装

运行需要 Python 3.12 或更新版本及可访问的 DeepSeek 视觉接口。所需 Python 库列于 `skills/image-case-retrieval/scripts/requirements.txt`。

```powershell
python -m pip install -r skills/image-case-retrieval/scripts/requirements.txt
```

作为 Codex Skill 使用时，将 `skills/image-case-retrieval` 整个目录复制到 Codex 的 `skills` 目录，然后调用 `$image-case-retrieval`。独立运行脚本不要求安装 Codex。

## 数据契约

```text
private-data/
├── E2E_1321cases_1w_data_v19.xlsx
├── ES_byFileType_user10_20261117170358.xlsx
├── .secrets/
│   └── deepseek.key
└── 图片/
    ├── 动物/
    ├── 建筑与场所/
    └── 其他类别/
```

文件名遵循已实现的数据适配器；其他数据集需先转换为同一格式，或单独修改适配器并创建新的实验目录。

用例工作簿使用第一个工作表，第一行为表头，从第二行读取：

| 列 | 内容 | 用途 |
|---|---|---|
| A | 非空且唯一的用例 ID | 贯穿计划、预测、矩阵和评分 |
| B | 非空自然语言查询 | 模型检索输入 |
| D | 已知正例 fid，可有多个 | 仅评分阶段读取 |

ES 工作簿各工作表的第一行是字段名。必需字段为 `fid`、`fileName`、`fileSize`；`shootingTime` 用于时间约束。`description`、`aiDescription`、`tagIds` 等可用于单独声明的元数据基线，主视觉流程不将其发送给模型。

fid 优先通过不区分大小写的文件名及字节大小匹配本地图片；找不到大小匹配时，允许唯一同名匹配。多重同名歧义保留为歧义，不任意选取一个文件。若一个 fid 对应多个已确认本地文件，命中任意一个即可。解码支持 JPEG、PNG、WebP、GIF、HEIC/HEIF、LIVP、ARW 和 DNG；GIF 使用静态帧，LIVP 提取首个可识别静态图，不评测视频动作。

## 凭据与模型

二选一：设置环境变量 `DEEPSEEK_API_KEY`，或把密钥保存到数据根目录的 `.secrets/deepseek.key`。不要把真实密钥写进命令示例、脚本或 Git 提交。

默认模型为 `deepseek-flash`；通过 `--model` 或 `DEEPSEEK_MODEL` 更改。默认 API 基址为 `https://api.deepseek.com`，可设置 `DEEPSEEK_BASE_URL`。替换模型或服务后应重新验证视觉输入、JSON 模式和输出格式，不沿用历史准确率。

## 单批端到端运行

以下命令在仓库根目录执行：

```powershell
python skills/image-case-retrieval/scripts/run.py --root ./private-data --out ./runs/batch-001 --offset 0 --limit 10 --workers 8 --max-calls 2000
python skills/image-case-retrieval/scripts/run.py --root ./private-data --out ./runs/batch-001 --offset 0 --limit 10 --stage evaluate
```

`--offset` 从零开始，`--limit` 为用例数量；因此 `--offset 10 --limit 10` 表示第 11–20 条。默认入口执行清点、fid 映射、目录规划、拼图筛选、单图复核、导出及预测封存。`evaluate` 独立读取答案并评分。`--max-calls` 限制本次进程中尚未缓存的模型逻辑调用数；单次逻辑调用的传输重试最多三次，因此该参数不是账单请求数或金额上限。

## 分批运行 100 条

建议沿用每批十条的规划方式，避免单次目录规划输出过长。

```powershell
foreach ($offset in 0,10,20,30,40,50,60,70,80,90) {
    $runOut = "./runs/batch-$offset"
    python skills/image-case-retrieval/scripts/run.py --root ./private-data --out $runOut --offset $offset --limit 10
    if ($LASTEXITCODE -ne 0) { throw "Inference failed at offset $offset" }
    python skills/image-case-retrieval/scripts/run.py --root ./private-data --out $runOut --offset $offset --limit 10 --stage evaluate
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed at offset $offset" }
}
```

这会产生新的实验结果，不会修改仓库内的历史基准。API 模型别名可能随时间变化，即使温度为零也不能保证与归档预测逐项一致。

## 续跑与分阶段操作

可选 `--stage prepare|screen|verify|export|evaluate`。发生可恢复故障后，用相同输入、参数及输出目录重新执行。成功响应和已完成图批会复用；未完成筛选或复核时禁止导出后直接评分。

工作簿哈希、图片清单、大小及修改时间用于检测输入变化。用例范围、模型、服务地址或拼图大小改变时，脚本要求新的输出目录；改变代码或提示词后，也应主动创建新的输出目录。不要把多个进程指向同一个输出目录。

缩略图按图片 ID 加锁，JPEG 和 JSON 通过临时文件原子替换。损坏派生图片保存在 `cache_quarantine/`。HTTP 400/401/402/403 会停止该 API 会话的新请求；已在执行的请求可能仍完成。传输或响应 JSON 失败最多尝试三次；错误消息不输出密钥或服务响应正文。

## 输出与解释

| 文件 | 内容 |
|---|---|
| `run_manifest.json` | 输入哈希与推理配置 |
| `inventory.json`、`images.csv` | 用例清单、图片 ID 与路径映射、排除文件 |
| `plan.json` | 每个查询的目录与必要条件 |
| `screen/`、`verify/`、`cache/` | 筛选、逐图判断及成功响应缓存 |
| `predictions.json`、`predictions_seal.json` | 状态、判断证据和文件哈希 |
| `pilot_matrix.csv` | 此批用例与全图库的矩阵 |
| `full_matrix.csv` | 全部用例结构，未运行用例留空 |
| `matches.csv`、`review.html` | 相关及待核实候选，含本地图片链接 |
| `evaluation.json` | 独立评分后的逐条及汇总指标 |

矩阵状态：`1` 为单图确认相关，`0` 为拼图筛选或单图复核未命中，`?` 为待核实，`E` 为解码失败，`S` 为目录未搜索，空白为该用例未运行。`0` 包含缩略图筛选结论，不是完整人工负例标注。Codex 对照矩阵仅记录正关联，空白同时包含未审和未标正，不能视为负例。

`review.html` 中的本地图片链接只适用于生成该文件的数据环境。发布报告应另行脱敏，不直接上传包含照片路径、caption 或完整 API 缓存的输出目录。

## 历史基准的复核范围

`python tools/verify_report.py` 使用公开匿名标签和预测集合，重新计算逐条与汇总成绩、元数据开发基线、矩阵状态及文件哈希。它不调用模型，也不读取私有数据。

持有原数据时，可使用 `tools/export_private_benchmark.py --root <数据根目录>` 重新导出已完成的私有实验；该工具适配已归档的 `image-matching/runs/pilot-v1` 和 `holdout-011-100-v1` 布局，不是通用推理入口。

历史冻结代码位于 `benchmarks/2026-11-17/frozen/`。其中仅将原机器默认数据根路径替换为通用默认值；原文件与公开副本的哈希和变换记录均见 `provenance.json`。历史批次协调器依赖原实验目录，新的数据运行应使用上文的发布入口。冻结方法、匿名预测复算与发布入口测试分别提供不同层面的复现证据。
