# ぽちゃガチョ！客服 AI

面向中国客服团队的日文玩家邮件自动处理系统。客服粘贴日文邮件，系统完成翻译、意图识别、知识库检索，并生成日文回复草稿，同步展示中文译文供客服审阅。支持在线管理知识库、导入新文件、调整检索参数。

---

## 功能概览

- **邮件即时翻译** — 日文邮件自动翻译为中文，客服无需日语能力即可理解内容
- **意图分类** — 自动识别问询类型，路由到对应知识库分区（分区列表动态扩展）
- **混合检索** — 向量检索 + BM25 双路并行，RRF 融合后重排，展示 Top-K 历史参考案例（含 JP/ZH 对照）
- **三档路由生成**
  - AUTO（≥阈值）：直接生成日文回复 + 复制按钮
  - REVIEW（中等置信度）：中文摘要 + 日文草稿，人工确认后发出
  - HUMAN（低置信度或多轮）：中文分析摘要 + 建议，建议人工回复
- **忠实度约束** — 生成时禁止编造参考案例中未出现的信息，AUTO 温度 0.1、REVIEW 温度 0.2
- **回复中文译文** — 生成的日文回复同步翻译为中文，方便客服校对
- **知识库管理** — 在线查看 / 编辑 / 删除 QA 条目，每次改动后自动热更新 FAISS+BM25 索引
- **文件导入** — 支持 Excel（Q/A 双列 / 单列段落）和 docx（Heading切分 / Q:A:标记），可自定义新分区名
- **检索参数配置** — 每个分区独立设置向量权重、BM25 权重、候选数、Top-K、路由阈值，UI 滑块实时调整
- **数据看板** — 可视化问询类型分布、版本热点、月度趋势、设备分布、置信度分布、路由等级分布、每日处理量

---

## 📹 演示视频

> 点击下方链接观看系统完整操作演示（时长约 4 分钟）
https://github.com/soo808/FatgooseGymAiEmailService/releases/download/demonstration-recording/FatgooseGymAiEmailService-demo-recording-20260421.mp4
---

## 技术栈

| 层次 | 技术 |
|------|------|
| 后端框架 | FastAPI + uvicorn（SSE 流式输出） |
| 向量模型 | intfloat/multilingual-e5-base（本地，~280MB，单例） |
| 向量索引 | FAISS IndexFlatIP（余弦相似度） |
| 稀疏检索 | rank_bm25（日文字符 bigram 分词） |
| 融合策略 | Reciprocal Rank Fusion（RRF，k=60） |
| LLM | DeepSeek API（deepseek-chat）— 分类 / 生成 / 翻译 |
| 知识库翻译 | 百度翻译 API（jp→zh，仅构建阶段） |
| 前端 | 纯 HTML + Chart.js CDN，无框架 |

---

## 目录结构

```
agenticRAG/
├── app/
│   ├── classifier.py       # DeepSeek 意图分类（动态读取分区列表）
│   ├── indexer.py          # FAISS+BM25 构建单一入口（model singleton）
│   ├── kb_manager.py       # 知识库 CRUD + 异步重建（asyncio.Lock）
│   ├── file_importer.py    # Excel / docx 文件解析（4 种模式）
│   ├── llm.py              # 生成 / 翻译函数（含忠实度约束）
│   ├── main.py             # FastAPI 路由 + SSE + 检索日志
│   ├── retriever.py        # HybridRetriever（config 驱动 + 热更新）
│   ├── routers/
│   │   └── kb.py           # /api/kb/* 知识库管理端点
│   └── static/
│       └── index.html      # 前端三 Tab 单页应用
├── scripts/
│   ├── build_kb.py         # 知识库构建（含百度翻译，支持断点续传）
│   └── analyze_kb.py       # 生成 analytics.json
├── kb/
│   ├── partitions/
│   │   ├── 不具合/         # qa_pairs.json · faiss.index · bm25.pkl
│   │   ├── 意見要望/
│   │   ├── 購入/
│   │   └── その他/
│   ├── retrieval_config.json   # per-partition 检索参数（可通过 UI 修改）
│   ├── retrieval_log.jsonl     # 检索质量日志（自动追加）
│   ├── analytics.json
│   └── translation_cache.json
├── .env                    # API 凭证（不提交 git）
├── requirements.txt
├── README.md
└── TECH.md
```

---

## 快速开始

### 1. 环境准备

Python 3.10 及以上（已在 3.14 验证）。

```bash
pip install -r requirements.txt
```

### 2. 配置凭证

编辑 `.env`：

```env
BAIDU_APPID=翻译 AppID
BAIDU_SECRET=翻译密钥
DEEPSEEK_API_KEY=sk-OpenAI密钥(本项目用的deepseek)
```

### 3. 构建知识库

```bash
# 完整构建（约 2000+ 条，含翻译，支持断点续传）
python scripts/build_kb.py --input D:/agenticRAG/RAW-emailQA-DATA.xlsx

# 快速验证（前 50 条，跳过翻译）
python scripts/build_kb.py --input RAW-emailQA-DATA-EXAMPLE.csv.xlsx --test --skip-translate
```

翻译进度通过 `kb/translation_cache.json` 缓存，中断后重新运行可从断点继续。

### 4. 生成看板数据

```bash
python scripts/analyze_kb.py
```

### 5. 启动服务

```bash
uvicorn app.main:app --reload --app-dir D:/agenticRAG
```

访问 [http://localhost:8000](http://localhost:8000)

---

## 知识库管理（在线操作）

服务启动后，切换到前端「📚 知识库管理」Tab：

| 操作 | 说明 |
|------|------|
| 分区筛选 | 点击分区按钮过滤，支持「全部」合并视图 |
| 编辑条目 | 点击卡片「编辑」→ 修改字段 → 保存后自动重建索引（SSE 进度条）|
| 删除条目 | 点击「删除」→ 确认后自动重建索引 |
| 导入文件 | 点击「⬆ 导入文件」→ 选文件 / 分区名 / 解析模式 → 自动写入并重建 |
| 检索配置 | 展开底部「⚙️ 检索参数配置」→ 调整滑块 → 保存（实时热更新，无需重启）|

### 文件导入解析模式

| 模式 | 适用场景 |
|------|---------|
| Excel — Q/A 双列 | 问答对格式，需指定 Q 列名 / A 列名 |
| Excel — 单列段落 | 规则/政策类文本，每行独立为一条 |
| docx — Q/A 标记行 | 正文以 `Q:` / `A:` 开头交替出现 |
| docx — Heading/段落 | Heading1/2 → 问题，后续段落 → 答案 |

导入新分区（如「游戏政策」）后，系统自动将分区名写入 `retrieval_config.json`，下次意图分类自动纳入该分区。

---

## 数据格式

源数据为 Excel 文件，必须包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `メールID` | str | 邮件线程唯一标识 |
| `メールID枝番` | int | 轮次编号（1=玩家，2=客服回复） |
| `本文` | str | 邮件正文 |

系统从枝番1正文中自动提取以下结构化字段（如果存在）：

- `お問い合わせ内容の種類` → inquiry_type（决定分区）
- `ご利用環境` → environment（iOS / Android / Google Play 等）
- `アプリバージョン` → app_version
- `問題が発生した日時` → date

---

## API 端点

| 端点 | 说明 |
|------|------|
| `POST /api/process` | SSE 流式处理邮件（主流程） |
| `GET /api/analytics` | 返回 analytics.json（看板数据） |
| `GET /api/metrics` | 返回检索质量指标（聚合 retrieval_log.jsonl）|
| `GET /api/health` | 健康检查，含分区列表和条目数 |
| `GET /api/kb/partitions` | 所有分区 + 条数 + 配置摘要 |
| `GET /api/kb/entries/{partition}` | 分页列表（`?page=1&page_size=20`）|
| `PUT /api/kb/entries/{partition}/{id}` | 更新条目字段（partial update）|
| `DELETE /api/kb/entries/{partition}/{id}` | 删除条目 |
| `POST /api/kb/rebuild/{partition}` | SSE 流：重建分区索引 |
| `POST /api/kb/import` | multipart 文件导入（SSE 进度流）|
| `GET /api/kb/config` | 读取 retrieval_config.json |
| `PUT /api/kb/config/{partition}` | 更新分区检索参数（热更新）|

---

## 翻译说明

| 阶段 | 翻译方式 | 原因 |
|------|---------|------|
| 知识库构建（`build_kb.py`） | 翻译 API | 批量翻译 2000+ 条，成本低，支持断点续传 |
| 运行时邮件翻译 | LLM API | 单次调用，需要上下文理解（游戏术语） |
| 运行时回复翻译 | LLM API | 同上 |
| 在线导入条目 | 留空（⚠️ 未翻译） | 导入后可手动填写，不影响向量检索 |

---

## 常见问题

**Q: `SSL UNEXPECTED_EOF` 错误**
A: 已通过 `http.client.HTTPSConnection` 直连解决（绕过系统代理 + OP_IGNORE_UNEXPECTED_EOF），无需修改。

**Q: 翻译出现 `[内容含敏感词，翻译已跳过]`**
A: 错误码 20003，该条 QA 对会保留占位符继续处理，不影响整体构建。

**Q: 首次运行很慢**
A: multilingual-e5-base 模型约 280MB，会在首次使用时自动从 HuggingFace 下载，之后本地缓存复用。进程内只加载一次（model singleton）。

**Q: 修改检索参数后需要重启服务吗？**
A: 不需要。前端「检索参数配置」面板保存后立即热更新，retriever 内的权重和阈值实时生效。

**Q: 导入新分区后意图分类能识别吗？**
A: 可以。分类器在每次调用时动态读取 `retrieval_config.json`，新分区写入 config 后下次分类自动纳入。
