# 技术方案简述

> 复盘文档 · 2026-04-11 · ぽちゃガチョ！客服 AI v3

---

## 一、背景与问题定义

**场景**：日本手游《ぽちゃガチョ！》的玩家邮件均为日文，由中国客服团队处理。核心矛盾：

1. 客服不懂日文，无法直接读懂玩家诉求
2. 回复必须用日文，且需保持标准客服语气
3. 问题类型繁杂（故障、建议、购买），处理难度差异大
4. 积累的历史 QA 对（2000+条）未被有效利用
5. 知识库维护困难，无法在线增删改查或导入新内容

**目标**：构建一套辅助系统，使中国客服无需日语能力，也能高效处理日文玩家邮件，并支持知识库的持续运营维护。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     构建阶段（离线）                      │
│                                                           │
│  RAW-emailQA-DATA.xlsx                                    │
│         ↓ build_kb.py                                     │
│  提取 QA 对 → 百度翻译 jp→zh → 分区（4类）               │
│         ↓ indexer.py（model singleton）                   │
│  multilingual-e5-base → FAISS IndexFlatIP                 │
│  bigram 分词 → BM25Okapi                                  │
│         ↓                                                 │
│  kb/partitions/<分区>/ {qa_pairs.json, faiss.index, bm25.pkl} │
│  kb/retrieval_config.json（per-partition 检索参数）        │
│                                                           │
│  analyze_kb.py → kb/analytics.json（看板数据）            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     推理阶段（在线）                      │
│                                                           │
│  客服输入日文邮件                                         │
│         ↓ asyncio.gather 并行                            │
│  ┌──────────────┐  ┌──────────────────────────────┐      │
│  │ DeepSeek 翻译 │  │ DeepSeek 意图分类             │      │
│  │ (jp→zh)      │  │ → 分区名（动态读 config）      │      │
│  └──────────────┘  └──────────────────────────────┘      │
│         ↓                                                 │
│  HybridRetriever.search(query, partition)                 │
│    ├─ 向量检索 Top-candidate_k（FAISS）                   │
│    ├─ BM25 检索 Top-candidate_k（bigram）                 │
│    ├─ RRF 融合（k=60）                                    │
│    └─ 重排（vec_weight×vs + rrf_weight×rs）→ Top-K       │
│         ↓                                                 │
│  per-partition 路由阈值（retrieval_config.json）          │
│    ≥auto_threshold  → AUTO：直接生成日文回复（temp=0.1）  │
│    ≥review_threshold → REVIEW：摘要 + 草稿（temp=0.2）   │
│    <review_threshold → HUMAN：摘要 + 建议（temp=0.3）    │
│         ↓ SSE 流式输出                                    │
│  DeepSeek 生成（忠实度约束提示词）                        │
│         ↓                                                 │
│  translate_reply → 日文回复中文译文                       │
│  BackgroundTask → retrieval_log.jsonl（检索质量日志）     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  知识库管理（在线）                       │
│                                                           │
│  前端「📚 知识库管理」Tab                                 │
│    ├─ CRUD：PUT/DELETE /api/kb/entries/{partition}/{id}  │
│    ├─ 每次 CRUD 后 → kb_manager.rebuild_index()          │
│    │    asyncio.to_thread(indexer.rebuild_partition_index)│
│    │    → 原子写入（.tmp → os.replace）                  │
│    │    → retriever.reload_partition()（热更新）         │
│    ├─ 文件导入：POST /api/kb/import（multipart SSE）      │
│    │    file_importer.parse_*() → add_entry × N → rebuild│
│    │    → add_partition_to_config()（新分区写入 config）  │
│    └─ 检索配置：PUT /api/kb/config/{partition}           │
│         → retriever.reload_config()（无需重启）          │
└─────────────────────────────────────────────────────────┘
```

---

## 三、关键设计决策

### 3.1 分区知识库 vs 单一知识库

**选择**：按 inquiry_type 建立独立分区（初始 4 个，可通过文件导入扩展）。

**理由**：故障类与建议类的语义空间差异大，单一索引会产生跨类干扰；分区后检索精度更高；分区可独立更新，互不影响；意图分类失败时降级到全分区搜索保底。

**代价**：分类错误导致检索目标分区错误（实测分类准确率 >95%）。

---

### 3.2 per-partition 检索参数

**选择**：`kb/retrieval_config.json` 统一存储各分区权重和阈值，UI 可在线调整。

**参数选择依据**：

| 分区 | vec/rrf | candidate_k | top_k | auto_t | review_t | 原因 |
|------|---------|-------------|-------|--------|----------|------|
| 不具合 | 0.60/0.40 | 30 | 5 | 0.88 | 0.70 | bug 报告含精确词（版本号/设备型号），BM25 权重提高；重复 bug 多，AUTO 阈值可适当放宽 |
| 意見要望 | 0.82/0.18 | 15 | 3 | 0.92 | 0.75 | 建议表达多样，语义向量权重高；个性化强，AUTO 阈值严格 |
| 購入 | 0.65/0.35 | 25 | 5 | 0.85 | 0.68 | 精确词（金额/订单）+ 情绪表达双重需求，平衡权重 |
| その他 | 0.75/0.25 | 20 | 5 | 0.90 | 0.70 | 兜底分区，保持均衡默认值 |

**热更新**：`retriever.reload_config()` 直接替换内存中的 config dict，无需重启服务。

---

### 3.3 检索层 vs 生成层的独立优化

**核心原则**：检索质量（召回率/精确率）和生成质量（忠实度/幻觉）是两个独立维度，不要混淆治理手段。

- **检索层问题**：调整 vec/rrf 权重、candidate_k、Top-K、分区参数
- **生成层问题**：修改 prompt 约束、降低温度

**具体措施（v3 新增）**：
```
AUTO/REVIEW system prompt 追加：
  ⚠️ 重要：回复内容只能基于上述参考案例中的实际信息。
  禁止添加参考案例中未出现的功能说明、版本说明、补偿政策或具体数值。
  如有不确定内容请说「確認後にご連絡いたします」，不得自行编造。

温度：AUTO → 0.1（收敛），REVIEW → 0.2，HUMAN → 0.3（保持，无检索参考）
```

---

### 3.4 知识库在线管理

**选择**：全量 CRUD + 每次操作后立即重建分区索引（asyncio.to_thread + asyncio.Lock）。

**理由**：
- 客服运营需要实时纠错（错误回答、过期信息），不能等离线重建
- 新版本/活动内容需要快速入库（文件导入 + 即时生效）
- 并发保护：同一分区一把 asyncio.Lock，防止并发重建文件损坏
- 原子写入：`.tmp` → `os.replace()`，防止半写状态被 FAISS 读取到

**代价**：大分区（1278 条）重建约需 15–30 秒，期间该分区索引锁定；小改动付出较大重建成本（可接受，频率低）。

---

### 3.5 Model Singleton（indexer.py）

**选择**：将 `SentenceTransformer` 实例提取为进程级单例（`app/indexer.py: get_model()`）。

**理由**：
- v2 中 `HybridRetriever.__init__` 和 `build_kb.py` 各自加载一次模型，合并后节省 ~280MB 内存
- `indexer.py` 同时提供 `rebuild_partition_index()` 和 `tokenize_jp()`，成为 build_kb / retriever / kb_manager 的共同依赖，避免代码重复

---

### 3.6 动态意图分类器

**选择**：分类器在每次调用时从 `retrieval_config.json` 读取分区列表和描述，构建提示词。

**理由**：
- 导入新分区（如「游戏政策」）后，`add_partition_to_config()` 将其写入 config，下次分类调用自动包含
- 失败时回退到硬编码列表，保证服务不中断
- 兜底分区（`その他`）始终放在列表末尾，作为默认分类

---

### 3.7 检索质量可观测性

**选择**：每次处理请求后（BackgroundTask）追加一行到 `retrieval_log.jsonl`，`/api/metrics` 聚合后返回。

**日志格式**：
```json
{"ts":"2026-04-11T10:00:00Z","partition":"不具合","confidence":87.3,"level":"REVIEW",
 "top1_score":0.873,"top5_scores":[0.873,0.845,0.812,0.790,0.768],"latency_ms":342}
```

**指标定义**（无 ground truth 时的代理指标）：
- **置信度分布**：top1_score 的分布，高置信度占比越高说明 KB 覆盖越好
- **路由等级分布**：AUTO 比例高说明系统可信度高，HUMAN 比例高说明 KB 或分类存在问题
- **score gap**：top1 与 top5 分数差，gap 大说明最优候选明确

---

### 3.8 文件导入与新分区扩展

**选择**：支持 4 种解析模式（Excel Q/A双列、Excel 单列段落、docx Q/A标记、docx Heading切分），导入后自动写入分区 config。

**设计要点**：
- 导入后 `question_zh`/`answer_zh` 留空，UI 显示「⚠️ 未翻译」badge（不影响检索，向量用 `question_jp` 编码）
- 新分区初始使用默认检索参数（vec=0.75/rrf=0.25/cand=20/top_k=5），可通过 UI 调整
- 新分区名写入 config 后，意图分类器和前端分区筛选栏在下次调用时自动更新

---

### 3.9 SSE 流式输出

**选择**：Server-Sent Events（单向流）而非 WebSocket（双向）。

**事件协议**（顺序）：
```
translate → intent → meta → section/token（循环）→ reply_zh → done
```

`translate` 和 `intent` 由 `asyncio.gather` 并行完成，通常在 2–3 秒内到达，之后才发出 `meta`，避免前端空白等待。知识库重建和文件导入也使用 SSE 流返回进度。

---

## 四、数据流与延迟分析

```
用户点击「处理邮件」
  ↓ ~0ms    POST /api/process
  ↓ ~2–4s   asyncio.gather(translate_email, classify_intent)
  ↓          → event: translate（前端显示邮件中文译文）
  ↓          → event: intent
  ↓ ~0.1s   HybridRetriever.search（本地推理，per-partition 参数）
  ↓          → event: meta（前端显示置信度 + Top-K 检索来源）
  ↓ 异步     BackgroundTask: 写 retrieval_log.jsonl（不阻塞主流程）
  ↓ ~1–3s   LLM 首 token（DeepSeek 流式，忠实度约束 prompt）
  ↓ 持续     → event: token（前端逐字显示）
  ↓ ~2s     translate_reply（非流式，等全文完成后）
  ↓          → event: reply_zh
  ↓          → event: done
```

总端到端延迟：约 8–15 秒（取决于邮件长度和 DeepSeek 响应速度）。

---

## 五、已知问题与局限

| 问题 | 影响 | 处理方式 |
|------|------|---------|
| 百度翻译敏感词过滤（error 20003） | 少量 QA 对译文为占位符 | 跳过并标记，不影响向量索引（用日文原文编码）|
| DeepSeek 余额不足（402） | 无法生成回复 | 前端提示充值；知识库原始回复仍可展示 |
| 意图分类误分 | 检索到错误分区 | 降级到全分区搜索保底 |
| 多轮邮件（含引用行 >） | 统一路由 HUMAN 档 | 当前仅识别轮次，未分析历史上下文 |
| 大分区重建耗时 | 1278条分区约 15–30 秒 | 期间索引锁定；小改动代价较高 |
| 在线导入条目无中文译文 | 检索结果展示不完整 | 显示「⚠️ 未翻译」badge，可手动填写 |

---

## 六、后续迭代方向

### 近期（可快速实现）

1. **Reranker 精排** — 在 Top-30 候选上叠加 cross-encoder（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`），替代当前的线性加权重排，预计 Recall@5 提升 5–10%

2. **多轮上下文处理** — 当邮件包含引用行（>）时，提取最新一轮内容和前置对话，分别送入检索和生成，而非直接标记为 HUMAN

3. **人工反馈回路** — 客服在前端标记「回复已采用 / 不合适」，写入 feedback.json，用于定期微调 prompt 或调整路由阈值

4. **在线条目翻译** — 导入后自动调用 DeepSeek 翻译 question/answer 的中文字段，而非留空

### 中期

5. **结构化元数据过滤** — 检索前先按 app_version 或 environment 预筛选（FAISS IDMap + filter），解决版本特定 bug 的精准召回问题

6. **知识库质量管理** — 建立 QA 对评分机制（被采用次数、客服满意度），低质量条目定期下线

7. **大分区增量更新** — 新增条目时改为 `faiss.index.add()` 而非全量重建（需维护 ID 映射），显著降低重建延迟

### 长期

8. **轻量级 fine-tune** — 积累足够反馈后，对 multilingual-e5-base 进行 domain-specific contrastive fine-tune

9. **自动化评估集** — 从历史数据中采样构建评估集（100条），衡量每次迭代后的 Recall@5 和路由准确率，防止退化

---

## 七、环境与依赖版本记录

| 包 | 版本 | 说明 |
|----|------|------|
| Python | 3.14 | 生产环境实测版本 |
| fastapi | ≥0.100.0 | SSE via StreamingResponse |
| sentence-transformers | ≥2.2.0 | multilingual-e5-base 加载 |
| faiss-cpu | ≥1.7.4 | IndexFlatIP |
| rank_bm25 | 0.2.2 | BM25Okapi |
| openai | ≥1.0.0 | DeepSeek OpenAI-compatible API |
| numpy | ≥1.24.0 | 向量运算 |
| python-docx | ≥1.1.0 | docx 文件解析（v3 新增）|
| python-multipart | ≥0.0.9 | multipart 文件上传（v3 新增）|

**注意**：Python 3.12+ 的 `ssl.OP_IGNORE_UNEXPECTED_EOF` 对百度翻译 API 的 TLS close_notify 问题至关重要，不可降级到标准 SSL 处理。

---

*文档维护：每次架构变更后更新第三、四节；迭代完成后将完成项从第六节移除并在第二节架构图中体现。*
