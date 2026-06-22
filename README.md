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
cp .env.example .env                   # 填入 MISTRAL / QDRANT / LANGSMITH key
python scripts/run_ingestion.py        # Day 1-2: fetch -> parse -> chunk
python scripts/build_index.py          # Day 3-4: embed (Mistral) -> 索引到 Qdrant
python scripts/query.py "What AI practices are prohibited?"   # Day 3-4: 纯检索
python scripts/ask.py   "What AI practices are prohibited?"   # Day 5: 检索→grade→作答/拒答
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
- `structure`：Top-4 全部命中 **Article 5**，带 `context_header` + 章节，可直接溯源引用，全10有约束力的正文。
- `baseline`：#1 命中正确内容但只是 `chunk 126`（无条款号）；#2-#4 落到 **Recital（非约束性前言）**，且无 metadata 可区分——印证了「丢结构」的代价。
- 越界问题（"chocolate cake"）得分 ~0.62 vs 在范围内 0.85+，分离明显，可作 Day 5 低置信度拒答的阈值依据。

### Future work：Hybrid 检索（dense + sparse）

纯 dense 检索对**精确术语/条款号**（"deployer"、"general-purpose AI model"、"Article 5(1)(h)"）的关键词匹配易漏，而这在法律问答里很关键。Qdrant + langchain-qdrant 原生支持 `RetrievalMode.HYBRID`（FastEmbed 稀疏向量，如 BM25/SPLADE），可把 dense 语义召回与 sparse 关键词召回融合。计划在 Day 8-9 作为评测表又一行（dense vs hybrid）评估收益后再决定接入——需加 `fastembed` 依赖并重建带稀疏向量的 collection。

## 编排与生成（Day 5）

用 **LangGraph** 把检索升级为完整问答闭环，核心是「检索不到/不相关就拒答、绝不编造法条」——法律问答刚需。

```
START → retrieve → grade ─(relevant)→ generate → END
                     └────(irrelevant)→ refuse  → END
```

- **grade 两层**（`src/generation/grade.py`）：
  1. **score 阈值**（`GRADE_MIN_SCORE=0.65`，纯函数 `score_gate`）：top hit 低于阈值直接拒答，**不花 LLM 调用**。阈值取自 Day 3-4 实测分离（在范围内 ≥0.72，越界 ~0.62）。
  2. **LLM 复核**（过阈值后）：Mistral 用 `with_structured_output` 二分判定 context 是否真能回答（CRAG/self-RAG 风格），输出 `relevant + reason`。
- **Grounded generation**（`src/generation/prompts.py` `ANSWER_PROMPT`）硬约束：① 只用提供的 context；② 每条结论标注 Article/Annex/Recital 号（取自 `context_header`）；③ recital 是非约束性材料，只有 recital 命中时可据其作答但须注明「非约束性」；④ context 确实无依据时**只输出哨兵 `INSUFFICIENT_CONTEXT`**（不让 LLM 自己复述拒答语）。
- **拒答确定性（两条路径都逐字）**：`refuse` 节点写死 `REFUSAL_TEXT`；generate 内 LLM 判不足时只吐哨兵，由纯函数 `finalize_answer` 映射成同一份 `REFUSAL_TEXT` 并置 `refused=True`。生成模型 `mistral-small-latest`、温度 0，均走 `src/generation/config.py`。
- **LangSmith tracing**：`.env` 配好 `LANGSMITH_*` 即自动上报——LangGraph 图 + 每次 ChatMistralAI 调用作为 trace 树；链外检索用 `@traceable` 包成 `retrieve` 子节点，可见召回 docs+score。实测拒答分支（~0.8s）显著快于作答分支（~3s），因短路了生成 LLM。

```bash
python scripts/ask.py "What AI practices are prohibited?"          # 命中 → 引用 Article 5 作答
python scripts/ask.py "how do I bake a chocolate cake"            # 越界 → score 阈值拦截、确定性拒答
python scripts/ask.py "definition of deployer" --show-context     # 边界 → 过阈值、LLM 复核后作答
```

### 踩坑记录：grade 放行 ≠ 能作答，导致拒答被误标

**现象**（query「What is trustworthy AI?」）：`grade=relevant`（top 0.824，召回全是 recital），但 `answer` 却是拒答语、`refused` 仍是 `False`，且拒答语被 LLM 截断（三句变两句）。

**根因**：拒答有**两条路径**——`refuse` 节点（确定性）与 generate 内 LLM 自行拒答（LLM 控制）。早期 `refused` 由「跑了哪个节点」决定，而非「实际输出是不是拒答」，所以 generate 内部的拒答被静默标成「成功作答」，且 LLM 复述 `REFUSAL_TEXT` 时不保证逐字。深层原因：AI Act 正文无 "trustworthy AI" 的约束性定义（只在 recital 出现），宽松的 grader 看到主题吻合就放行，严格的 answerer 却找不到可引用的正式定义——两个 LLM 在回答不同问题。

**解法（sentinel 方案）**：answerer 判不足时只输出哨兵 `INSUFFICIENT_CONTEXT`，纯函数 `finalize_answer` 检测哨兵 → 替换成规范 `REFUSAL_TEXT` 并置 `refused=True`。这样①拒答语**保证逐字**、②`refused` **反映真实输出**、③最终拒答权交给最严格、最接近输出的 answerer（grade 退为省 token 的廉价预筛）。同时放宽 prompt 允许「据 recital 作答但注明非约束性」，修掉过度拒答——现在该 query 能正确据 **Recital 27** 作答并标注「non-binding」。Day 8-9 评测与 Day 10-11 审计日志依赖 `refused` 准确，此修复是前提。

### 已知技术缺陷（审计记录）- TODO

**🟡 中等（建议 Day 6-7 前处理）**

- 【已解决】**分数阈值硬编码假设 Cosine，但 `config.DISTANCE` 是"权威配置"**。`Retriever.search` 的 `s >= min_score`、`score_gate` 的 `hits[0].score >= 0.65`、`GRADE_MIN_SCORE` 全部假设「相似度越大越相关」。Day 3-4 让 `DISTANCE` 变成 config 驱动并在建库生效，但**检索侧打分语义没跟着走**：一旦改成 `"Euclid"`，Qdrant 返回的是距离（越小越好），所有 gate 逻辑**静默反转**且无报错。**解法**：把"分数方向 + 阈值标定的距离度量"集中成单一权威——`config.SCORE_CALIBRATED_DISTANCE`（= 标定阈值所用的度量）与纯函数 `config.assert_score_threshold_semantics()`，并在每个阈值比较处（`Retriever.search` 的 `min_score`、`score_gate`、`select_answer_hits`）调用。阈值的「方向」与「量纲」都与 Cosine 绑定，故一旦 `DISTANCE` 偏离标定度量即**显式报错**（提示重新标定并翻转比较方向），不再静默反转。测试覆盖 Cosine 通过 / Euclid 抛错两路径。
- 【已解决】**图里网络调用零异常处理**。`retrieve`/`grade`/`generate` 裸调 Qdrant/Mistral，任意超时/429/5xx 异常直接冒泡出 `answer_question`。CLI 下只是难看的 trace，但 **Day 6-7 FastAPI 下是未捕获 500 + 泄露内部栈**（对合规叙事减分）。且 `grade` 的 `llm_grade` 抛错会连已通过 score-gate 的结果一起崩。**解法**：分两类处理——① **硬依赖**（`retrieve`/`generate`）失败 → 包成单一受控异常 `generation.errors.PipelineError`（带 `stage` + 调用方安全文案，原异常 `raise ... from e` 链在内部日志、不外泄栈），供 Day 6-7 API 层统一捕获映射；**刻意不**把宕机伪装成拒答（`refused` 须保持权威，宕机是 error 非 refusal）。② **软复核**（`grade` 的 `llm_grade`）失败 → 优雅降级为「score-gate 通过即 relevant」（带降级原因），不因一次复核抖动崩掉整请求；score-gate 已拒的结果不会被降级复活。测试覆盖三条路径（retrieve/generate 抛 `PipelineError` 且链住原异常、grade 降级、grade 仍按 score-gate 拒答）。

**🟢 低级别（非 bug，留待对应阶段）**

- **两个独立分数阈值**：`Retriever.search(min_score=)`（query.py 用）与 `GRADE_MIN_SCORE`（图里 `score_gate` 用）是两个常量，调一个易误以为两个都动。
- 【已解决】**`generate` 把全部 5 个 hit 不加区分喂给生成**：top hit 过阈值后，其余弱块（哪怕 ~0.4）也进 context，稀释答案质量（引用约束兜底，非错误）。可在生成前加每-hit 软阈值。
- **`finalize_answer` 用子串匹配哨兵**：正经答案正文若恰含 `INSUFFICIENT_CONTEXT` token 会误判拒答（概率极低）。可收紧为 `strip()` 后相等/startswith。
- 【需要考虑结果质量，暂时不做】**grade 与 generate 各发一份全量 context** → 每次问答约 2× token，纯成本；grade 复核可用更短摘要。
- **`grade="relevant"` 且 `refused=True` 是合法状态**（sentinel 修复后 answerer 有最终拒答权，覆盖 grade）。这是预期且正确的——Day 8-9 评测/Day 10-11 审计应把 **`refused` 当权威、`grade` 当建议**，勿用 grade 算拒答率。

**📘 已知/已处理**：HF tokenizer warning（已用 `HF_HUB_OFFLINE` 修）；Hybrid 检索（见上 future work）；Day 6-7 并发——`get_embeddings`/`get_chat_llm`/`_get_retriever`/`build_graph` 均 lru_cache 单例，FastAPI 多请求共享，只读调用一般安全但上线前值得压测。

## 目录

```
data/raw/         原始 HTML 快照 + fetch 元数据（提交）
data/processed/   解析与切分产物（gitignore，可复现）
src/ingestion/    schema / fetch / parse / chunk
src/retrieval/    config / embeddings / index / retriever（Qdrant + Mistral）
src/generation/   config / llm / prompts / grade / graph（LangGraph + LangSmith）
scripts/          run_ingestion.py · build_index.py · query.py · ask.py
tests/            pytest sanity 断言（ingestion + retrieval + generation，不联网）
```

## 路线图

- [x] **Day 1-2**：数据摄取与预处理
- [x] **Day 3-4**：Embedding（Mistral）+ Qdrant 索引 + 向量检索（rerank 插槽预留）
- [x] **Day 5**：LangGraph 编排（retrieve→grade→generate/refuse）+ grounded 生成 + LangSmith tracing
- [ ] Day 6-7：FastAPI + Docker
- [ ] Day 8-9：RAGAS 评测 + MLflow
- [ ] Day 10-11：合规层（来源追溯 / PII / 审计日志）
- [ ] Day 12：Demo + README
