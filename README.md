# EU AI Act 问答助手 — RAG（建设中）

面向受监管领域的检索增强问答系统，语料为 **EU AI Act（Regulation (EU) 2024/1689）**。

## 语料版本声明

| 项 | 值 |
| --- | --- |
| 法规 | Regulation (EU) 2024/1689（Artificial Intelligence Act） |
| CELEX | `32024R1689` |
| 来源 | EUR-Lex 官方 HTML（OJ L 2024/1689） |
| 抓取日期 | 见 `data/raw/fetch_metadata.json` |
| Digital Omnibus 修订 | **未纳入**（2026-05 Council doc 9247/26 引入的 Art. 4a/60a/75a-75e 等） |

> v1 采用 OJ 基准文本，未合并 Digital Omnibus 修订
> 抓取的原始 HTML 快照提交在 `data/raw/`，作为语料版本锁定（reproducibility）。

## 快速开始

```bash
conda env create -f environment.yml
conda activate aiact-rag
python scripts/run_ingestion.py        # fetch -> parse -> chunk
pytest -q                              # sanity 检查
```

产物（`data/processed/`，由原始 HTML 可复现，故 gitignore）：
- `units.jsonl` — 结构化法律单元（recital / article / annex，带层级 metadata）
- `chunks_baseline.jsonl` — 固定切分（structure-blind 基线）
- `chunks_structure.jsonl` — 结构感知切分（保留条款完整性 + 可追溯 metadata）

## 两种 Chunking 策略（对比是评测的第一行素材）

| 策略 | 切法 | metadata | 定位 |
| --- | --- | --- | --- |
| `baseline` | 全文拉平后固定 ~512 token + 64 overlap | 仅 source/version/index | 基线（固定size） |
| `structure` | 每个 recital/article/annex 一个 chunk；超长再 sub-split | unit_type / number / title / chapter / section / sub_index | 保住条款完整性与按条款检索 |

token 数用 `tiktoken` cl100k 仅作尺寸代理（Mistral 分词器不同，不影响切分控制）。

=== chunking comparison (tiktoken cl100k tokens) ===
strategy    chunks    mean  median    p95
-----------------------------------------
baseline       301   355.3     393    510
structure      402   265.1   227.0    509

### baseline 的一个隐性缺陷：overlap 并非均匀生效

`baseline` 名义上带 `chunk_overlap=64`，但实测 300 对相邻 chunk 中**只有 18 对真正共享重叠，282 对是零重叠硬切**。原因是 `RecursiveCharacterTextSplitter` 的 overlap 以"切分片段"为单位回退：单元间用 `\n\n` 切出的是**整条 recital/article**（200–500 token，远大于 64），换块时整条被弹出 → 边界硬切无重叠；只有**超长单元内部**被递归切成句子级小片段时，才留得下 ≈64 token 的重叠尾巴。

即"块块有缓冲"是错觉——重叠只在长条款内部生效，条款与条款之间是零重叠硬切，短条款的语境因此易被邻居挤入同一块又被硬边界截断。这正是 `structure`（按条款对齐边界）有望在 context precision 上胜出的机制性原因，留待 Day 8-9 RAGAS 验证。

## 目录

```
data/raw/         原始 HTML 快照 + fetch 元数据（提交）
data/processed/   解析与切分产物（gitignore，可复现）
src/ingestion/    schema / fetch / parse / chunk
scripts/          run_ingestion.py 编排入口
tests/            pytest sanity 断言
```

## 路线图

- [x] **Day 1-2**：数据摄取与预处理（本阶段）
- [ ] Day 3-4：Embedding（Mistral）+ Qdrant 索引 + 检索/rerank
- [ ] Day 5：LangGraph 编排 + LangSmith tracing
- [ ] Day 6-7：FastAPI + Docker
- [ ] Day 8-9：RAGAS 评测 + MLflow
- [ ] Day 10-11：合规层（来源追溯 / PII / 审计日志）
- [ ] Day 12：Demo + README
