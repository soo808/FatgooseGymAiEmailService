# 技术方案简述

> 复盘文档 · 2026-04-11 · ぽちゃガチョ！客服 AI v2

---

## 一、背景与问题定义

**场景**：日本手游《ぽちゃガチョ！》的玩家邮件均为日文，由中国客服团队处理。核心矛盾：

1. 客服不懂日文，无法直接读懂玩家诉求
2. 回复必须用日文，且需保持标准客服语气
3. 问题类型繁杂（故障、建议、购买），处理难度差异大
4. 积累的历史 QA 对（2000+条）未被有效利用

**目标**：构建一套辅助系统，使中国客服无需日语能力，也能高效处理日文玩家邮件。

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     构建阶段（离线）                      │
│                                                           │
│  RAW-emailQA-DATA.xlsx                                    │
│         ↓ build_kb.py                                     │
│  提取 QA 对 → 百度翻译 jp→zh → 分区（4类）               │
│         ↓                                                 │
│  各分区独立构建：                                         │
│    multilingual-e5-base → FAISS IndexFlatIP               │
│    bigram 分词 → BM25Okapi                                │
│         ↓                                                 │
│  kb/partitions/<分区>/ {qa_pairs.json, faiss.index, bm25.pkl} │
│                                                           │
│  analyze_kb.py → kb/analytics.json（看板数据）            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     推理阶段（在线）                      │
│                                                           │
│  客服输入日文邮件                                         │
│         ↓ asyncio.gather 并行                            │
│  ┌──────────────┐  ┌──────────────────────┐              │
│  │ DeepSeek 翻译 │  │ DeepSeek 意图分类     │              │
│  │ (jp→zh)      │  │ → 分区名              │              │
│  └──────────────┘  └──────────────────────┘              │
│         ↓                                                 │
│  HybridRetriever.search(query, partition)                 │
│    ├─ 向量检索 Top-20（FAISS）                            │
│    ├─ BM25 检索 Top-20（bigram）                          │
│    ├─ RRF 融合（k=60）                                    │
│    └─ 重排（0.7×vec + 0.3×rrf_norm）→ Top-5             │
│         ↓                                                 │
│  置信度路由（final_score × 100）                          │
│    ≥90%  → AUTO：直接生成日文回复                         │
│    70–90% → REVIEW：摘要 + 草稿                           │
│    <70%   → HUMAN：摘要 + 建议                            │
│         ↓ SSE 流式输出                                    │
│  DeepSeek 生成（参考 Top-3 案例）                         │
│         ↓                                                 │
│  translate_reply → 日文回复中文译文                       │
└─────────────────────────────────────────────────────────┘
```

---

## 三、关键设计决策

### 3.1 分区知识库 vs 单一知识库

**选择**：按 inquiry_type 建立 4 个独立分区（不具合 / 意見要望 / 購入 / その他）。

**理由**：
- 故障类与建议类的语义空间差异大，单一索引会产生跨类干扰
- 分区后每个索引规模缩小，检索精度更高
- 方便按分区独立更新，不影响其他分区
- 意图分类失败时降级到全分区搜索，不中断流程

**代价**：分类错误会导致检索目标分区错误（实测分类准确率 >95%）。

---

### 3.2 混合检索（向量 + BM25）

**选择**：两路并行检索，各取 Top-20，RRF 融合后重排取 Top-5。

**理由**：
- 纯向量检索对关键词匹配弱（如版本号、设备型号等精确词）
- BM25 对语义泛化弱（同义表达无法召回）
- 两路互补，RRF 融合简单有效，无需额外模型

**评分公式**：
```
final_score = 0.7 × vec_score(余弦相似度) + 0.3 × rrf_normalized
```

向量分权重更高，因 multilingual-e5-base 在日文跨语义场景表现稳定；BM25 作为关键词补偿。

---

### 3.3 百度翻译（构建）vs DeepSeek（运行时）

**构建阶段**：百度翻译批量翻译 2000+ 条 QA 对。
- 成本极低（免费额度 5 万字符/月，超出约 0.05 元/千字符）
- 断点续传缓存，中断可继续
- 质量足够用于知识库展示，不要求文学水准

**运行时**：DeepSeek 翻译单封邮件 + 翻译生成的日文回复。
- 需要理解游戏上下文和专业术语
- 单次调用，延迟可接受（约 2–4 秒）
- DeepSeek 在日文→中文翻译质量优于百度

---

### 3.4 SSE 流式输出

**选择**：Server-Sent Events（单向流）而非 WebSocket（双向）。

**理由**：
- 场景为服务端推送，无需客户端反向通信
- SSE 基于 HTTP，无额外握手，FastAPI 原生支持 `StreamingResponse`
- 前端无需 WebSocket 库，原生 ReadableStream API 即可处理

**事件协议**（顺序）：
```
translate → intent → meta → section/token（循环）→ reply_zh → done
```

`translate` 和 `intent` 由 `asyncio.gather` 并行完成，通常在 2–3 秒内到达，之后才发出 `meta`，避免前端空白等待。

---

### 3.5 Embedding 模型选型

**选择**：intfloat/multilingual-e5-base（768维，~280MB）

**理由**：
- 明确支持日文 + 中文多语言检索
- 使用 "passage:"/"query:" 前缀设计，区分检索侧和文档侧
- 本地推理，无 API 费用，延迟可控
- 精度与 large 版本差距可接受（base 已够用）

**候选替代**：paraphrase-multilingual-mpnet-base-v2（无前缀设计，稍弱）；text-embedding-3-small（需 API，成本高）。

---

### 3.6 前端无框架设计

**选择**：纯 HTML + 原生 JS，Chart.js CDN，无 React/Vue。

**理由**：
- 系统为内部工具，维护人员不一定熟悉前端框架
- 功能固定，无复杂状态管理需求
- 单文件部署，便于拷贝和分发
- Chart.js 满足看板需求，无需引入更重的数据可视化库

---

## 四、数据流与延迟分析

```
用户点击「处理邮件」
  ↓ ~0ms    POST /api/process
  ↓ ~2–4s   asyncio.gather(translate_email, classify_intent)
  ↓          → event: translate（前端显示邮件中文译文）
  ↓          → event: intent
  ↓ ~0.1s   HybridRetriever.search（本地推理）
  ↓          → event: meta（前端显示置信度 + Top-5 检索来源）
  ↓ ~1–3s   LLM 首 token（DeepSeek 流式）
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
| 版本/日期字段稀疏 | 元数据标签大量为空 | 字段可选，不影响核心检索 |

---

## 六、后续迭代方向

### 近期（可快速实现）

1. **Reranker 精排** — 在 Top-30 候选上叠加 cross-encoder（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`），替代当前的线性加权重排，预计 Recall@5 提升 5–10%

2. **多轮上下文处理** — 当邮件包含引用行（>）时，提取最新一轮内容和前置对话，分别送入检索和生成，而非直接标记为 HUMAN

3. **人工反馈回路** — 客服在前端标记「回复已采用 / 不合适」，写入 feedback.json，用于定期微调 prompt 或调整路由阈值

4. **知识库增量更新** — 新积累的 QA 对无需全量重建，对指定分区执行 `faiss.index.add()` 并更新 BM25，同步更新 qa_pairs.json

### 中期

5. **结构化元数据过滤** — 检索前先按 app_version 或 environment 预筛选（FAISS IDMap + filter），解决版本特定 bug 的精准召回问题

6. **知识库质量管理** — 建立 QA 对评分机制（被采用次数、客服满意度），低质量条目定期下线，避免噪声累积

7. **多语言扩展** — 将 inquiry_type 分类 prompt 改为多语言版，支持英文玩家邮件接入同一套系统

### 长期

8. **轻量级 fine-tune** — 积累足够反馈后，对 multilingual-e5-base 进行 domain-specific contrastive fine-tune，提升游戏领域日文的检索质量

9. **自动化测试集** — 从历史数据中采样构建评估集（100条），衡量每次迭代后的 Recall@5 和路由准确率，防止退化

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

**注意**：Python 3.12+ 的 `ssl.OP_IGNORE_UNEXPECTED_EOF` 对百度翻译 API 的 TLS close_notify 问题至关重要，不可降级到标准 SSL 处理。

---

*文档维护：每次架构变更后更新第三、四节；迭代完成后将完成项从第六节移除并在第二节架构图中体现。*
