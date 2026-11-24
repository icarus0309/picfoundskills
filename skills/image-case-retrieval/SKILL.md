---
name: image-case-retrieval
description: 根据自然语言用例检索本地图片，融合已有描述、OCR、拍摄时间与地点元信息和视觉检索，使用 DeepSeek 规划、Qwen 复核，并在预测封存后按 fid 标签评测。适用于批量图片匹配和召回优化。
---

# 本地图片用例检索

将查询工作簿与本地图片库建立关联，输出相关图片、证据、矩阵和可复查页面。扩展说明见 [数据与运行文档](https://github.com/icarus0309/picfoundskills/blob/main/docs/usage.md)；独立安装的核心接口以本文件和 `scripts/` 为准。

## 高权重检索

要求按“高权重预期fid”评分、扩大找图覆盖并处罚多找时，使用 `weighted_retrieval.py` 与 `score_weighted.py`。评分按表头识别高权重列，本地 v19 在 F 列。每张返回的清单外图片均计入精确率分母。默认历史 V1/V2 报表仍按 D 列复现。[高权重检索命令、证据来源与评测口径](references/high-weight.md)

## 输入与入口

数据根目录包含：

- `E2E_1321cases_1w_data_v19.xlsx`：首个工作表，A 列为用例 ID、B 列为查询、D 列为评测用已知正例 fid。
- `ES_byFileType_user10_20261117170358.xlsx`：至少包含 `fid`、`fileName`、`fileSize`；时间条件使用 `shootingTime`。
- `图片/`：按类别划分的图片目录；目录内 `.txt` 可提供目录说明。

时间同时参考 `tagIds` 的年份和 `shootingTime`；地点使用 `country`、`province`、`city`、`business_terms`、`place_terms`。字段缺失或冲突单列，不能默认画面外观能够证明拍摄地点。

先确定查询范围，通常每批十条。DeepSeek 文本规划凭据使用 `<root>/.secrets/deepseek.key` 或 `--llm-key-file`。Qwen 视觉配置文件包含 `base_url`、`model`、`key_file`，密钥单独存放于该文件所指向的私有文件，不能写入脚本、报告或版本库。配置示例：

```json
{"base_url":"https://<workspace>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1","model":"qwen3.8-27b","key_file":"qwen.key"}
```

```powershell
python -m pip install -r scripts/requirements.txt
python scripts/hybrid_run.py --root <数据根目录> --out <输出目录> --offset 0 --limit 10 --vlm-config <私有配置文件>
python scripts/score_completed.py --root <数据根目录> --out <输出目录>
```

`--offset` 从零开始；`--limit` 是数量。非连续用例可通过 `--cases-file` 提供 `cases` 数组，每项保留原始 `id` 和 `query`；必须与 A/B 列一致，数量与 `--limit` 相同。默认文本模型 `deepseek-flash`，视觉模型由私有配置指定。替换服务后先验证视觉输入和 JSON 兼容性。

`run.py` 保留 V1 单批入口；`hybrid_run.py` 在同一 VLM 下生成 V1 视觉策略和 V2 混合策略的成对预测。两种方法共享已完成的视觉分支，不能当作独立双倍调用计算成本。

## 检索约束

1. 推理只读取用例 A/B 列；D 列及其他答案列不进入目录规划、筛选、复核或候选规则。
2. 清点图片与非图片。HEIC/HEIF 由 pillow-heif 解码，LIVP 提取静态图，ARW/DNG 使用 rawpy。解码失败保留 `E`，不能转为负例。
3. 视觉分支根据目录和说明规划候选目录，以 12 图拼图筛选，再使用长边不超过 1200 像素的独立图片复核。Qwen 复核请求每次一张图，两种方法使用相同设置。
4. 含视觉条件的正式相关结果必须经过单图复核；纯结构化条件的例外见第 6 项。模糊细节、时间缺失或冲突保留 `uncertain`。年份不得由像素猜测；元数据本身也可能需要核查。
5. V1 视觉基线不读取已有图片语义描述；V2 对全库描述/OCR执行主题词与同义词精确检索，并加入中文双字 BM25 候选，取与视觉候选的并集。输入差异须在报告中披露。
6. 没有额外视觉条件且所有明确结构化年份/地点条件匹配时，可直接元信息确认；有对象、颜色、关系或未解析的时间条件时仍需复核。地点或年份明确不等于必须在画面出现文字招牌。
7. V2 保留基线正关联，因此正例召回非下降是并集性质，不代表精确率得到保证。不采用固定 top-k 数量截断；未搜索、描述缺失和模型筛选仍可漏召回。

## 续跑与失败处理

`hybrid_run.py` 仅推理与封存；`score_completed.py` 检查全部批次完成及预测哈希后，才读取 D 列进行成对评分。不得在其他批次预测未完成时提前评分或根据答案调整条件。

使用相同输出目录续跑；输入、用例范围、模型、端点或推理规则改变时使用新目录。冻结协议和代码快照在输出根目录。仅修复执行基础设施时，也必须保留修改前快照与变更说明，不能覆盖历史证据。不要并发启动两个主运行进程写入同一输出目录。

缓存保存成功响应、模型返回名称、用量与图像摘要。临时 JSON/JPEG 原子替换，缩略图按稳定图片 ID 加锁。鉴权、余额及普通非重试 HTTP 错误停止该 API 会话；传输错误最多三次。图像审核 `data_inspection_failed` 单独处理：不再提交被拦截图片，保守登记受影响图像并标待核实，其他请求继续。不能改图绕过拦截，也不能把拦截视为不相关。

单图响应先严格核对图片和查询编号；去除说明、输入回显及完全相同的重复记录，不改变判断状态。校正重试两次仍无有效结论时，登记 `schema_abstentions/` 并标待核实，不能从冲突结果中挑选有利判断。两组使用相同处理，并在报告中披露数量。

服务商明确报告 JSON 输出生成中止时，记录 `provider_generation_errors/` 并有限重试；不能将所有 HTTP 400 都视为可重试。目录规划的编号或目录名不合法时，只按原始输入校正，不手工替模型选择目录。

对于客户端自行构造的图像 data URI，服务若明确返回 URI 解析错误，可保持原图和查询不变有限重试，记录 `provider_media_errors/`。耗尽后单列为待核实，不执行编号校正重试，也不视为内容审核拦截。`runtime_diagnostics/` 保存已脱敏的服务错误，属于本地诊断材料，不直接发布。

拼图响应在普通编号校正后仍不合法、导致 `screen` 中止时，可执行下列辅助脚本，再用相同主命令续跑。它只处理已有失败响应，使用更清楚的编号与格子映射重筛，保存原缓存和修复日志；不会重做已成功图批或改动查询标准。加 `--watch` 可与主进程并行等待失败图批。

```powershell
python scripts/repair_screen_identifiers.py --root <数据根目录> --out <输出目录> --vlm-config <私有配置文件>
```

## 评测与交付

预测封存后，评分阶段按文件名及字节大小建立 fid 映射。歧义或缺失映射单列。D 列允许多个逗号、分号或换行分隔的 fid；每个用例内去重。

- 用例命中率：至少一个预期 fid 出现在返回集合中的用例占比。
- 已知 fid 覆盖率：各用例已命中的预期 fid 数之和除以预期 fid 数之和。
- 同时报告返回关联数、待核实数、本地无法映射数量及输入范围。
- 只有正例标签时，不能据此计算真实 precision 或全部相关图片的完整 recall。
- 开发集与新增验证集分开统计；查询近义重复、人工候选辅助方式及审图覆盖范围必须披露。
- 计分后发现标签冲突应保留原始成绩，单独输出诊断，不删除困难项或回写答案提高分数。

混合运行根目录包含 `protocol.json`、`inventory.json`、`corpus_metadata.json`、`frozen_code/`、`prediction_completion.json` 和服务商拦截登记。各 `batch-*/visual-v1/`、`batch-*/hybrid-v2/` 保存预测、矩阵、封存与后续评分；`scoring_summary.json` 汇总两种方法。共享 `cache/` 位于批次目录，`metadata_candidates.json` 记录文本和结构化候选来源。

矩阵：`1`=视觉复核或明确结构化条件确认相关；`0`=拼图筛选或单图复核未命中；`?`=待核实；`E`=解码失败；`S`=目录未搜索；空白=该用例未运行。`0` 不能等同于精细人工负例，`S` 和空白不能改成 `0`。
