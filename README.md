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
cp .env.example .env                   # 填入 MISTRAL_API_KEY / QDRANT_URL / QDRANT_API_KEY
python scripts/run_ingestion.py        # Day 1-2: fetch -> parse -> chunk
python scripts/build_index.py          # Day 3-4: embed (Mistral) -> 索引到 Qdrant
python scripts/query.py "What AI practices are prohibited?"   # 检索
pytest -q                              # sanity 检查（不联网）
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

## 检索（Day 3-4）

- **Embedding**：Mistral `mistral-embed`（1024 维，全欧洲栈）。
- **向量库**：Qdrant Cloud，两套 chunk 各建一个 collection（`aiact_baseline` / `aiact_structure`），便于 Day 8-9 直接对比检索质量。点 id 用 `uuid5(chunk_id)` 确定性生成，重建即 upsert 不产生重复。维度/距离由 `config.EMBED_DIM/DISTANCE` 显式驱动并在建库时校验。
- **幂等**：以 chunk 文件的 **sha256 内容指纹**（记于 `data/processed/.index_meta.json`）判断是否需要重建——内容变了即使条数不变也会自动重建，避免留下旧向量；`--recreate` 强制重建。
- **检索**：向量召回 top-k（默认 20）→ **rerank 插槽（本版为 identity passthrough，已预留 Cohere）** → top-n（默认 5）。支持 `unit_type` + 条款号区间（`number_int`）组合过滤与 `min_score` 阈值（Qdrant 对 payload 字段 `unit_type/number_int` 自动建索引）。

```bash
python scripts/build_index.py                          # 索引两套（--recreate 强制重建）
python scripts/query.py "prohibited AI practices" --strategy structure
python scripts/query.py "high-risk" --unit-type article --number-min 6 --number-max 15 --min-score 0.8
```

**structure vs baseline（同一问题「What are the prohibited AI practices?」）**：
- `structure`：Top-4 全部命中 **Article 5**，带 `context_header` + 章节，可直接溯源引用，全为有约束力的正文。
- `baseline`：#1 命中正确内容但只是 `chunk 126`（无条款号）；#2-#4 落到 **Recital（非约束性前言）**，且无 metadata 可区分——印证了「丢结构」的代价。
- 越界问题（"chocolate cake"）得分 ~0.62 vs 在范围内 0.85+，分离明显，可作 Day 5 低置信度拒答的阈值依据。

### Future work：Hybrid 检索（dense + sparse）

纯 dense 检索对**精确术语/条款号**（"deployer"、"general-purpose AI model"、"Article 5(1)(h)"）的关键词匹配易漏，而这在法律问答里很关键。Qdrant + langchain-qdrant 原生支持 `RetrievalMode.HYBRID`（FastEmbed 稀疏向量，如 BM25/SPLADE），可把 dense 语义召回与 sparse 关键词召回融合。计划在 Day 8-9 作为评测表又一行（dense vs hybrid）评估收益后再决定接入——需加 `fastembed` 依赖并重建带稀疏向量的 collection。

## 目录

```
data/raw/         原始 HTML 快照 + fetch 元数据（提交）
data/processed/   解析与切分产物（gitignore，可复现）
src/ingestion/    schema / fetch / parse / chunk
src/retrieval/    config / embeddings / index / retriever（Qdrant + Mistral）
scripts/          run_ingestion.py · build_index.py · query.py
tests/            pytest sanity 断言（ingestion + retrieval，不联网）
```

## 路线图

- [x] **Day 1-2**：数据摄取与预处理
- [x] **Day 3-4**：Embedding（Mistral）+ Qdrant 索引 + 向量检索（rerank 插槽预留）
- [ ] Day 5：LangGraph 编排 + LangSmith tracing
- [ ] Day 6-7：FastAPI + Docker
- [ ] Day 8-9：RAGAS 评测 + MLflow
- [ ] Day 10-11：合规层（来源追溯 / PII / 审计日志）
- [ ] Day 12：Demo + README
