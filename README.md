# PicFound Skills

融合本地图片已有描述、OCR、拍摄时间、地点元信息与视觉检索的可复用 Skill。默认使用 DeepSeek 规划检索条件，使用 Qwen 复核图像，并输出可续跑、可追溯的用例—图片关联矩阵。

## 最新验证：时间与地点查询

从未参与历史评测的后续查询中，根据 A/B 列筛选全部时间或具名地点条件，共 **162 条**：时间 56 条、地点 95 条、组合 11 条。图库包含 **10,489 张静态图片**。

| 方法 | 用例命中 | 用例命中率 | 已知fid微平均覆盖率 | 返回正关联 |
|---|---:|---:|---:|---:|
| V1 视觉策略＋Qwen | 125/160 | 78.1% | 22.8% | 3,285 |
| V2 描述/元信息/视觉融合＋Qwen | 154/160 | 96.3% | 46.1% | 7,420 |

用例命中只要求找到至少一个预期 fid；已知 fid 覆盖率更能反映是否找回更多已列正例，但仍不是完整语义召回率或真实精确率。V2 保留 V1 正关联，新增结果须由明确结构化条件或多模态复核确认；召回非下降是并集设计的性质，不代表精确率必然提高。

两组使用相同 Qwen 后端。历史 DeepSeek 在前 100 条上的 96% 命中率属于不同查询集，不能直接作为本次提升幅度。服务商图像审核拦截单列为待核实，不算负例。

- [完整时间地点对比报告](docs/evaluation-time-place.md)
- [逐条结果](benchmarks/2026-11-17-time-place/data/cases.csv)、[Excel 对比表](benchmarks/2026-11-17-time-place/comparison.xlsx)
- [V1 矩阵](benchmarks/2026-11-17-time-place/data/baseline_matrix.csv)、[V2 矩阵](benchmarks/2026-11-17-time-place/data/hybrid_matrix.csv)
- [匿名标签与预测集合](benchmarks/2026-11-17-time-place/data/benchmark.json)、[协议与溯源](benchmarks/2026-11-17-time-place/provenance.json)

## 使用方法

- [Skill 入口](skills/image-case-retrieval/SKILL.md)
- [安装与运行说明](docs/usage.md)

在仓库根目录安装依赖，将数据放入 `private-data/`。文本规划密钥保存为 `private-data/.secrets/deepseek.key`；视觉配置通过私有 JSON 指定端点、模型与独立密钥文件，不将真实凭据写入仓库。

```powershell
python -m pip install -r skills/image-case-retrieval/scripts/requirements.txt
python skills/image-case-retrieval/scripts/hybrid_run.py --root ./private-data --out ./runs/demo --limit 10 --vlm-config ./private-data/.secrets/qwen.json
python skills/image-case-retrieval/scripts/score_completed.py --root ./private-data --out ./runs/demo
```

V2 同时使用 `tagIds` 年份、`shootingTime`、`country`、`province`、`city`、`business_terms` 和 `place_terms`。只有纯时间/地点条件且元信息明确匹配时才直接确认；存在视觉对象或属性条件时仍需图像复核。已有描述检索保留全部主题词命中与合格 BM25 候选，不采用固定 top-k 数量截断。

## 离线复核

无需私有照片或 API 密钥：

```powershell
python tools/verify_report.py
python tools/verify_hybrid_report.py
python -m unittest discover -s skills/image-case-retrieval/tests -v
```

公开材料使用稳定图片 ID 和 SHA-256 转换后的 fid，保留评分所需映射，不包含原图、原始 fid、图片文件名、完整描述、GPS 坐标、私有端点或密钥。读者可以独立复算指标；语义和元信息真实性审核仍需要原始数据。

## 历史实验

前 100 条原始验证：[完整报告](docs/evaluation-report.md)、[历史基准数据](benchmarks/2026-11-17)。该实验主视觉流程使用 DeepSeek，累计用例命中 96/100、已知 fid 覆盖 3,075/6,256（49.2%）；Codex 元数据辅助直接审图样本命中 94/100。两种历史方法的输入和覆盖范围不同。

原实验、冻结推理代码和当前成对验证均保留，未使用计分后的标签反向修改预测。
