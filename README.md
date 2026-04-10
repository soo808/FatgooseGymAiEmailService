# ぽちゃガチョ！客服 AI

面向中国客服团队的日文玩家邮件自动处理系统。客服粘贴日文邮件，系统完成翻译、意图识别、知识库检索，并生成日文回复草稿，同步展示中文译文供客服审阅。

---

## 功能概览

- **邮件即时翻译** — 日文邮件自动翻译为中文，客服无需日语能力即可理解内容
- **意图分类** — 自动识别问询类型（不具合 / 意見要望 / 購入 / その他），路由到对应知识库分区
- **混合检索** — 向量检索 + BM25 双路并行，RRF 融合后重排，展示 Top-5 历史参考案例（含 JP/ZH 对照）
- **三档路由生成**
  - AUTO（≥90%）：直接生成日文回复 + 复制按钮
  - REVIEW（70–90%）：中文摘要 + 日文草稿，人工确认后发出
  - HUMAN（<70% 或多轮）：中文分析摘要 + 建议，建议人工回复
- **回复中文译文** — 生成的日文回复同步翻译为中文，方便客服校对
- **数据看板** — 可视化问询类型分布、版本热点、月度趋势、设备分布

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 后端框架 | FastAPI + uvicorn（SSE 流式输出） |
| 向量模型 | intfloat/multilingual-e5-base（本地，~280MB） |
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
│   ├── classifier.py       # DeepSeek 意图分类
│   ├── llm.py              # 生成 / 翻译函数
│   ├── main.py             # FastAPI 路由 + SSE
│   ├── retriever.py        # HybridRetriever
│   └── static/
│       └── index.html      # 前端单页应用
├── scripts/
│   ├── build_kb.py         # 知识库构建（含翻译）
│   └── analyze_kb.py       # 生成 analytics.json
├── kb/
│   ├── partitions/
│   │   ├── 不具合/         # qa_pairs.json · faiss.index · bm25.pkl
│   │   ├── 意見要望/
│   │   ├── 購入/
│   │   └── その他/
│   ├── analytics.json
│   └── translation_cache.json
├── .env                    # API 凭证（不提交 git）
├── requirements.txt
├── README.md
└── TECH.md                 # 技术方案与迭代规划
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
BAIDU_APPID=你的百度翻译 AppID
BAIDU_SECRET=你的百度翻译密钥
DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
```

- 百度翻译：在 [fanyi-api.baidu.com](https://fanyi-api.baidu.com) 注册，开通「通用文本翻译API」服务
- DeepSeek：在 [platform.deepseek.com](https://platform.deepseek.com) 注册并获取 API Key

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

## 翻译说明

| 阶段 | 翻译方式 | 原因 |
|------|---------|------|
| 知识库构建（`build_kb.py`） | 百度翻译 API | 批量翻译 2000+ 条，成本低，支持断点续传 |
| 运行时邮件翻译 | DeepSeek API | 单次调用，需要上下文理解（游戏术语） |
| 运行时回复翻译 | DeepSeek API | 同上 |

百度翻译免费额度为每月 5 万字符（标准版），建议在控制台开启按量付费后使用完整数据集。

---

## 常见问题

**Q: `SSL UNEXPECTED_EOF` 错误**
A: 已通过 `http.client.HTTPSConnection` 直连解决（绕过系统代理 + OP_IGNORE_UNEXPECTED_EOF），无需修改。

**Q: 翻译出现 `[内容含敏感词，翻译已跳过]`**
A: 百度错误码 20003，该条 QA 对会保留占位符继续处理，不影响整体构建。

**Q: DeepSeek 402 余额不足**
A: 前端会显示提示。前往 platform.deepseek.com 充值后重试即可恢复生成功能。

**Q: 首次运行很慢**
A: multilingual-e5-base 模型约 280MB，会在首次使用时自动从 HuggingFace 下载，之后本地缓存复用。
