# PicFound Skills

将自然语言用例与本地图片库匹配的可复用 Skill。流程使用 DeepSeek 规划检索目录、筛选拼图候选并逐图复核，输出用例与图片关联矩阵，同时支持缓存续跑和基于 fid 标签的独立评分。

在包含 **1,321 条用例、10,489 张静态图片**的数据上，已完成前 100 条验证。按“至少返回一个预期 fid 即命中”的定义，Skill 累计命中率为 **96%**，Codex 元数据辅助直接审图样本为 **94%**。Skill 的已知 fid 覆盖率为 **49.2%**；用例命中率不表示已经找全所有相关图片。

| 范围 | DeepSeek Skill | Codex 辅助审图样本 |
|---|---:|---:|
| 开发集 1–10 | 9/10，90.0% | 10/10，100.0% |
| 新增验证集 11–100 | 87/90，96.7% | 84/90，93.3% |
| 累计 1–100 | 96/100，96.0% | 94/100，94.0% |

两种方法的输入及覆盖范围不同：Skill 不读取已有图片语义描述；Codex 候选使用 ES 描述预筛，新增 90 条实际检查 267 张不同图片、334 次用例—图片组合。因此这些结果属于工作流对照，不构成相同输入条件下的模型能力排名。前十条用于开发，新增查询也包含近义复述。

## 文档与测试证据

- [完整测试对比报告](docs/evaluation-report.md)：数据、方法、指标、逐项对照、失败分析、用量及局限。
- [安装与运行](docs/usage.md)：数据格式、凭据、分批执行、续跑及输出解释。
- [Skill 入口](skills/image-case-retrieval/SKILL.md)。
- [逐条结果 CSV](benchmarks/2026-11-17/data/cases_100.csv) 与 [Excel 对比报告](benchmarks/2026-11-17/comparison-100.xlsx)。
- [Skill 矩阵](benchmarks/2026-11-17/data/skill_matrix_100.csv) 与 [Codex 正关联矩阵](benchmarks/2026-11-17/data/codex_matrix_100.csv)，均为 100×10,489。
- [可独立计分的脱敏数据](benchmarks/2026-11-17/data/benchmark.json)、[实验溯源](benchmarks/2026-11-17/provenance.json) 及 [历史推理代码归档](benchmarks/2026-11-17/frozen)。

公开数据使用稳定图片 ID 和 SHA-256 转换后的 fid，不包含原始照片、原始 fid、文件名、已有图片描述或 API 密钥。读者无需访问私有图库，即可复核报告中的命中数与矩阵计数；重新执行视觉推理则需要原始数据及有效 API 凭据。

## 快速开始

推荐 Python 3.12 或更新版本。以下命令在仓库根目录执行。

```powershell
python -m pip install -r skills/image-case-retrieval/scripts/requirements.txt
```

将数据放入 `private-data/`，并通过环境变量 `DEEPSEEK_API_KEY` 配置凭据。先运行十条，再独立计分：

```powershell
python skills/image-case-retrieval/scripts/run.py --root ./private-data --out ./runs/demo --offset 0 --limit 10
python skills/image-case-retrieval/scripts/run.py --root ./private-data --out ./runs/demo --offset 0 --limit 10 --stage evaluate
```

默认推理入口会导出并封存预测，评分入口随后读取 D 列。相同参数及输出目录可用于续跑；输入或推理配置改变时使用新的输出目录。

## 离线复核

无需 API 密钥或原始照片：

```powershell
python tools/verify_report.py
python -m unittest discover -s skills/image-case-retrieval/tests -v
```

验证脚本重新计算逐条及汇总指标，检查匿名标签映射、两份矩阵和基准文件校验和。自动化测试覆盖端到端导出与评分、答案隔离、续跑、输入变化检测、并发缓存以及 API 失败处理。GitHub Actions 在 Windows 和 Linux 上配置了同一组离线检查。

发布入口包含通用路径、原子缓存写入、鉴权失败停止和缺失导出步骤等工程修复。**96% / 94% 来自已封存的历史实验，并非对发布版重新调用模型所得的成绩。**具体差异及复现范围见完整报告。
