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
| `structure` | 每个 recital/article/annex 一个 chunk；超长再 sub-split | unit_type / number / **number_int** / title / chapter / section / sub_index / **context_header** | 保住条款完整性与按条款检索 |

token 数用 `tiktoken` cl100k 仅作尺寸代理（Mistral 分词器不同，不影响切分控制）。

- **`number_int`**：条款号的数值形式（annex 罗马数字也转 int），供 Day 3-4 按条款号区间过滤/排序（字符串 `'10' < '2'` 会错乱）。
- **`context_header`**：每个 structure chunk 的文本前置 `"Article 6 — Classification…"` 这类自含前缀（同时存入 metadata），让只看文本、看不到 metadata 的 embedding 模型也知道碎片归属。已预留 token 预算，chunk 仍 ≤512。


```
=== chunking comparison (tiktoken cl100k tokens) ===
strategy    chunks    mean  median    p95
-----------------------------------------
baseline       301   355.3     393    510
structure      408   270.5   233.5    508
```

### baseline 的一个隐性缺陷：overlap 并非均匀生效

`baseline` 名义上带 `chunk_overlap=64`，但实测 300 对相邻 chunk 中**只有 18 对真正共享重叠，282 对是零重叠硬切**。原因是 chunk_overlap（重叠视窗）与 semantic boundary（语义边界，如 \n\n）之间的底层算法博弈导致的静默失效；当文本中出现长度超过重叠上限的巨大自然段落时，`RecursiveCharacterTextSplitter` 为了不破坏该段落的完整性，只能被迫放弃相邻 chunk 之间的 overlap，从而在长文本的切分中留下不可预知的 context 断层。

即"块块有缓冲"是错觉——重叠只在长条款内部生效，条款与条款之间是零重叠硬切，短条款的语境因此易被邻居挤入同一块又被硬边界截断。这正是 `structure`（按条款对齐边界）有望在 context precision 上胜出的机制性原因，留待 Day 8-9 RAGAS 验证。

### 已知限制（设计取舍，留待后续阶段）

- **baseline 无条款级 metadata**（仅 `chunk_index/source_url/version`）。这是对照实验的本意——baseline 故意丢结构、无法做条款级溯源。因此 **Day 10-11 合规层的 source attribution 只能建立在 `structure` 集上**，不要在 baseline 上做条款引用；评测时也应预期 baseline 在 attribution 维度天然为 0。
- **缺段落级（paragraph）粒度**。`structure` 的 chunk 知道是 Article 6，但不区分 6(1)/6(2)。chunk 文本内保留了 `1./2.` 内联编号、可事后恢复，但 metadata 未拆到子段。若 Day 10-11 需要把引用精确到 `Art 6(2)`，再补段落级抽取即可。

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
