---
name: image-case-retrieval
description: 根据自然语言用例检索本地图片，使用 DeepSeek 规划目录、拼图筛选和单图复核，并在预测封存后按 fid 标签评测。适用于批量图片匹配、标注候选生成和检索基准构建。
---

# 本地图片用例检索

将查询工作簿与本地图片库建立关联，输出相关图片、证据、矩阵和可复查页面。扩展说明见 [数据与运行文档](https://github.com/icarus0309/picfoundskills/blob/main/docs/usage.md)；独立安装的核心接口以本文件和 `scripts/` 为准。

## 输入与入口

数据根目录包含：

- `E2E_1321cases_1w_data_v19.xlsx`：首个工作表，A 列为用例 ID、B 列为查询、D 列为评测用已知正例 fid。
- `ES_byFileType_user10_20261117170358.xlsx`：至少包含 `fid`、`fileName`、`fileSize`；时间条件使用 `shootingTime`。
- `图片/`：按类别划分的图片目录；目录内 `.txt` 可提供目录说明。

先确认查询范围，通常每批十条。凭据使用 `DEEPSEEK_API_KEY` 或 `<root>/.secrets/deepseek.key`，不写入提示词、脚本、报告或版本库。

```powershell
python -m pip install -r scripts/requirements.txt
python scripts/run.py --root <数据根目录> --out <输出目录> --offset 0 --limit 10
python scripts/run.py --root <数据根目录> --out <输出目录> --offset 0 --limit 10 --stage evaluate
```

`--offset` 从零开始；`--limit` 是数量。默认模型 `deepseek-flash`，可用 `--model` 更改；默认 API 地址为 `https://api.deepseek.com`，可通过 `DEEPSEEK_BASE_URL` 更改。替换服务时须先验证其视觉输入及 JSON 响应兼容性。

## 检索约束

1. 推理只读取用例 A/B 列；D 列及其他答案列不进入目录规划、筛选、复核或候选规则。
2. 清点图片与非图片。HEIC/HEIF 由 pillow-heif 解码，LIVP 提取静态图，ARW/DNG 使用 rawpy。解码失败保留 `E`，不能转为负例。
3. 根据目录名、说明和查询规划候选目录。相关目录按 12 图拼图筛选，再将候选以长边不超过 1200 像素的独立图片复核，每请求最多四张。
4. 正式相关结果必须经过单图复核。模糊细节、时间缺失或冲突保留 `uncertain`。年份不得由像素猜测；元数据本身也可能需要核查。
5. 主视觉方法不使用已有图片 caption 或语义标签。若加入元数据检索，应声明为新的方法版本并分开评测。
6. 不用固定 top-k 冒充全部相关图片；目录未搜索与拼图未选中仍可能造成漏召回。

## 续跑与失败处理

`run.py` 默认完成清点、映射、规划、筛选、复核、导出与预测封存，不自动读取 D 列评分。随后运行 `--stage evaluate`。

单阶段支持 `prepare|screen|verify|export|evaluate`。使用相同输出目录续跑；输入表、图片清单、用例范围、模型、API 地址或拼图大小改变时新建输出目录。不要并发启动两个写入同一输出目录的进程。

缓存保存成功响应、模型返回名称、用量与图像摘要。临时 JSON/JPEG 原子替换，缩略图按稳定图片 ID 加锁；损坏派生缓存转存到 `cache_quarantine/`。HTTP 400/401/402/403 停止该 API 会话的新请求；传输和响应解析错误最多尝试三次。失败的图批不能当成负例，也不能跳过未完成阶段直接评分。

## 评测与交付

预测封存后，评分阶段按文件名及字节大小建立 fid 映射。歧义或缺失映射单列。D 列允许多个逗号、分号或换行分隔的 fid；每个用例内去重。

- 用例命中率：至少一个预期 fid 出现在返回集合中的用例占比。
- 已知 fid 覆盖率：各用例已命中的预期 fid 数之和除以预期 fid 数之和。
- 同时报告返回关联数、待核实数、本地无法映射数量及输入范围。
- 只有正例标签时，不能据此计算真实 precision 或全部相关图片的完整 recall。
- 开发集与新增验证集分开统计；查询近义重复、人工候选辅助方式及审图覆盖范围必须披露。
- 计分后发现标签冲突应保留原始成绩，单独输出诊断，不删除困难项或回写答案提高分数。

输出包括 `predictions.json`、`predictions_seal.json`、`evaluation.json`、`pilot_matrix.csv`、`full_matrix.csv`、`images.csv`、`matches.csv`、`review.html`。

矩阵：`1`=单图确认相关；`0`=拼图筛选或单图复核未命中；`?`=待核实；`E`=解码失败；`S`=目录未搜索；空白=该用例未运行。`0` 不能等同于精细人工负例，`S` 和空白不能改成 `0`。
