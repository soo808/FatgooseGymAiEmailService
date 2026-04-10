# RAG v2 — 多分区知识库 + 混合检索 + 看板 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将单一知识库升级为按问询类型分区的多 KB 架构，引入 BM25+向量混合检索 + RRF 融合 + 重排序（Top-5），重构前端为「邮件翻译 | 生成回复」双栏布局 + 检索来源面板 + 数据看板。

**Architecture:** `build_kb.py` 按 inquiry_type 建立 4 个独立分区（各含 FAISS + BM25 索引）；推理时 DeepSeek 先做意图分类确定分区，再对该分区并行执行向量/BM25 双路检索，RRF 融合取 Top-30，按向量相似度+元数据匹配打分重排为 Top-5；前端双栏展示译文与回复，下方展示重排后的 Top-5 来源卡片，Tab 切换至数据看板（Chart.js）。

**Tech Stack:** Python 3.14, FastAPI, FAISS, `rank_bm25`, `sentence-transformers` (multilingual-e5-base), DeepSeek API (分类+生成+翻译), Chart.js CDN

---

## 文件变更清单

| 文件 | 动作 | 职责 |
|------|------|------|
| `scripts/build_kb.py` | 修改 | 新增 date/version 提取、分区逻辑、BM25 索引构建 |
| `scripts/analyze_kb.py` | 新建 | 从分区 QA 对计算 analytics.json，供看板使用 |
| `app/classifier.py` | 新建 | 意图分类（返回分区名），非流式 DeepSeek 调用 |
| `app/retriever.py` | 重写 | HybridRetriever：加载所有分区，向量+BM25+RRF+重排 |
| `app/llm.py` | 修改 | 新增 `translate_email()`，更新生成 prompt 使用 Top-5 |
| `app/main.py` | 修改 | 新 SSE 事件协议，`GET /api/analytics`，`POST /api/translate` |
| `app/static/index.html` | 重写 | 双栏布局 + Top-5 来源卡片 + 数据看板 Tab |
| `requirements.txt` | 修改 | 新增 `rank_bm25>=0.2.2` |

### 生成的 KB 目录结构（运行新 build_kb.py 后）
```
kb/
├── partitions/
│   ├── 不具合/          qa_pairs.json · faiss.index · bm25.pkl
│   ├── 意見要望/         qa_pairs.json · faiss.index · bm25.pkl
│   ├── 購入/            qa_pairs.json · faiss.index · bm25.pkl
│   └── その他/          qa_pairs.json · faiss.index · bm25.pkl
└── analytics.json
```

---

## Task 1: 扩展 build_kb.py — 提取 date/version，按 inquiry_type 分区构建索引

**Files:**
- Modify: `scripts/build_kb.py`
- Modify: `requirements.txt`

### 分区映射逻辑（inquiry_type → partition name）
```python
PARTITION_MAP = {
    "不具合":    "不具合",
    "ご意見":    "意見要望",
    "意見":      "意見要望",
    "購入":      "購入",
}

def get_partition(inquiry_type: str) -> str:
    for key, val in PARTITION_MAP.items():
        if key in inquiry_type:
            return val
    return "その他"
```

- [ ] **Step 1: 在 `extract_question_fields` 中新增 date 和 app_version 提取**

在现有函数（`scripts/build_kb.py:88`）里，在 `environment` 提取之后追加：

```python
# 问题发生日期（格式: 2024-03-31 23:07）
m = re.search(r'問題が発生した日時\s*[：:]\s*(\d{4}-\d{2}-\d{2})', text)
date_str = m.group(1) if m else ""

# App 版本
m = re.search(r'アプリバージョン\s*[：:]\s*([\d.]+)', text)
app_version = m.group(1).strip() if m else ""

return question, inquiry_type, environment, date_str, app_version
```

同时将函数签名的 return 类型更新为 `tuple[str, str, str, str, str]`，所有调用处补上两个新变量。

- [ ] **Step 2: 更新 `extract_qa_pairs` 使用新签名，QA 对 dict 加入新字段**

在 `scripts/build_kb.py:138` 的 `extract_qa_pairs` 里：
```python
question, inquiry_type, environment, date_str, app_version = \
    extract_question_fields(_safe_str(turns[1]))

qa_pairs.append({
    "id": str(mail_id),
    "question_jp": question,
    "answer_jp": answer,
    "question_zh": "",
    "answer_zh": "",
    "inquiry_type": inquiry_type,
    "environment": environment,
    "app_version": app_version,
    "date": date_str,
    "total_turns": total_turns,
    "partition": get_partition(inquiry_type),  # ← 新增
})
```

- [ ] **Step 3: 在 `requirements.txt` 追加依赖**

```
rank_bm25>=0.2.2
```

安装：`pip install rank_bm25`

- [ ] **Step 4: 在 `main()` 中替换单一索引构建为分区构建**

删除原来的 `build_index(qa_pairs)` 调用，替换为：

```python
# ── 按分区构建索引 ──
from collections import defaultdict
partitions: dict[str, list[dict]] = defaultdict(list)
for qa in qa_pairs:
    partitions[qa["partition"]].append(qa)

print(f"\n[分区] 共 {len(partitions)} 个分区：")
for name, items in partitions.items():
    print(f"  {name}: {len(items)} 条")

build_partitioned_index(partitions)
```

- [ ] **Step 5: 实现 `build_partitioned_index`（替换原 `build_index`）**

在 `scripts/build_kb.py` 中，删除旧 `build_index` 函数，添加：

```python
def tokenize_jp(text: str) -> list[str]:
    """日文字符 bigram 分词，适用于 BM25"""
    text = re.sub(r'\s+', '', text)
    chars = list(text)
    bigrams = [chars[i] + chars[i+1] for i in range(len(chars) - 1)]
    return chars + bigrams


def build_partitioned_index(partitions: dict[str, list[dict]]):
    """为每个分区分别构建 FAISS + BM25 索引"""
    try:
        import faiss
        import pickle
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}")
        sys.exit(1)

    print("\n[向量] 加载 intfloat/multilingual-e5-base ...")
    model = SentenceTransformer('intfloat/multilingual-e5-base')

    part_dir = KB_DIR / "partitions"
    part_dir.mkdir(parents=True, exist_ok=True)

    for name, qa_list in partitions.items():
        out_dir = part_dir / name
        out_dir.mkdir(exist_ok=True)

        # 保存 QA JSON
        with open(out_dir / "qa_pairs.json", 'w', encoding='utf-8') as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=2)

        questions = [qa["question_jp"] for qa in qa_list]

        # FAISS
        embeddings = model.encode(
            ["passage: " + q for q in questions],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(out_dir / "faiss.index"))

        # BM25
        tokenized = [tokenize_jp(q) for q in questions]
        bm25 = BM25Okapi(tokenized)
        with open(out_dir / "bm25.pkl", 'wb') as f:
            pickle.dump(bm25, f)

        print(f"  [{name}] {len(qa_list)} 条 → FAISS {index.ntotal} + BM25 OK")

    print(f"\n[完成] 分区索引保存至 kb/partitions/")
```

- [ ] **Step 6: 验证**

```bash
python scripts/build_kb.py --input D:/agenticRAG/RAW-emailQA-DATA-EXAMPLE.csv.xlsx --test --skip-translate
```

预期输出含：
```
[分区] 共 N 个分区：
  不具合: XX 条
  意見要望: XX 条
  ...
[完成] 分区索引保存至 kb/partitions/
```

并确认 `D:/agenticRAG/kb/partitions/` 下各子目录含 3 个文件。

- [ ] **Step 7: 用完整已翻译数据重建（断点续传）**

```bash
python scripts/build_kb.py --input D:/agenticRAG/RAW-emailQA-DATA.xlsx
```

翻译阶段因有缓存会极快完成，只重建索引部分。

---

## Task 2: 新建 scripts/analyze_kb.py — 生成 analytics.json

**Files:**
- Create: `scripts/analyze_kb.py`

analytics.json 结构：
```json
{
  "total": 2201,
  "by_partition": {"不具合": 1400, "意見要望": 600, ...},
  "inquiry_type_dist": {"不具合について": 1380, ...},
  "environment_dist": {"Google Play": 900, "iOS": 1100, "": 201},
  "version_dist": {"1.1.4": 200, "1.2.0": 350, ...},
  "date_monthly": {"2024-01": 80, "2024-02": 120, ...},
  "version_x_partition": {"1.1.4": {"不具合": 150, "意見要望": 40}, ...}
}
```

- [ ] **Step 1: 创建文件**

```python
#!/usr/bin/env python
"""analyze_kb.py — 从分区知识库计算统计数据，输出 kb/analytics.json"""

import json
from collections import defaultdict
from pathlib import Path

KB_DIR = Path(__file__).parent.parent / "kb"
PARTS_DIR = KB_DIR / "partitions"


def load_all_qa() -> list[dict]:
    pairs = []
    for part_dir in PARTS_DIR.iterdir():
        json_path = part_dir / "qa_pairs.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                pairs.extend(json.load(f))
    return pairs


def count(items: list[str]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for x in items:
        c[x or "不明"] += 1
    return dict(sorted(c.items(), key=lambda kv: kv[1], reverse=True))


def main():
    qa_pairs = load_all_qa()
    if not qa_pairs:
        print("[错误] 未找到分区数据，请先运行 build_kb.py")
        return

    # 按分区统计
    by_partition = count([qa["partition"] for qa in qa_pairs])

    # 问询类型分布
    inquiry_dist = count([qa["inquiry_type"] for qa in qa_pairs])

    # 使用环境分布
    env_dist = count([qa["environment"] for qa in qa_pairs])

    # 版本分布（只取主版本号 X.Y）
    def normalize_version(v: str) -> str:
        if not v:
            return "不明"
        parts = v.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else v
    version_dist = count([normalize_version(qa["app_version"]) for qa in qa_pairs])

    # 按月统计（date 字段为 YYYY-MM-DD）
    monthly: dict[str, int] = defaultdict(int)
    for qa in qa_pairs:
        d = qa.get("date", "")
        month = d[:7] if len(d) >= 7 else "不明"
        monthly[month] += 1
    date_monthly = dict(sorted(monthly.items()))

    # 版本 × 分区 交叉分析
    vxp: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for qa in qa_pairs:
        v = normalize_version(qa.get("app_version", ""))
        p = qa.get("partition", "その他")
        vxp[v][p] += 1
    # 只保留 Top-10 版本
    top_versions = list(version_dist.keys())[:10]
    version_x_partition = {v: dict(vxp[v]) for v in top_versions if v in vxp}

    analytics = {
        "total": len(qa_pairs),
        "by_partition": by_partition,
        "inquiry_type_dist": inquiry_dist,
        "environment_dist": env_dist,
        "version_dist": version_dist,
        "date_monthly": date_monthly,
        "version_x_partition": version_x_partition,
    }

    out_path = KB_DIR / "analytics.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(analytics, f, ensure_ascii=False, indent=2)
    print(f"✓ analytics.json 已保存 → {out_path}")
    print(f"  总计: {len(qa_pairs)} 条 | 分区: {list(by_partition.keys())}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 运行并检查输出**

```bash
python scripts/analyze_kb.py
```

预期：打印 `✓ analytics.json 已保存`，检查 `kb/analytics.json` 内各字段非空。

---

## Task 3: 新建 app/classifier.py — 意图分类

**Files:**
- Create: `app/classifier.py`

- [ ] **Step 1: 创建文件**

```python
"""
classifier.py — 用 DeepSeek 对输入邮件做意图分类，返回分区名
分区: 不具合 | 意見要望 | 購入 | その他
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

PARTITIONS = ["不具合", "意見要望", "購入", "その他"]

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
    return _client


async def classify_intent(email_text: str) -> str:
    """
    返回分区名: "不具合" | "意見要望" | "購入" | "その他"
    失败时返回 "その他"（不中断主流程）
    """
    prompt = f"""以下は日本のゲームサポートに届いたメールです。
内容を読んで、最も当てはまるカテゴリを1つだけ選んでください。

カテゴリ:
- 不具合: バグ、クラッシュ、データ消失、動作不良
- 意見要望: 改善提案、新機能要望、感想
- 購入: 課金、アイテム未反映、返金
- その他: 上記以外

メール:
{email_text[:800]}

回答は上記4つのカテゴリのいずれか1単語のみ。"""

    try:
        resp = await _get_client().chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        raw = resp.choices[0].message.content.strip()
        for p in PARTITIONS:
            if p in raw:
                return p
        return "その他"
    except Exception:
        return "その他"
```

- [ ] **Step 2: 快速验证（可在 Python REPL 中运行）**

```python
import asyncio
from app.classifier import classify_intent

test = "アプリを最新にしたらゲームが起動しなくなりました。"
result = asyncio.run(classify_intent(test))
assert result == "不具合", f"got {result}"
print("OK:", result)
```

---

## Task 4: 重写 app/retriever.py — HybridRetriever

**Files:**
- Rewrite: `app/retriever.py`

检索流程：
```
query
  ↓ 向量检索 Top-20  ┐
  ↓ BM25 检索 Top-20 ┘→ RRF 融合 Top-30
                        ↓ 向量分+元数据 Boost 重排
                        ↓ Top-5
```

- [ ] **Step 1: 完整替换 retriever.py**

```python
"""
retriever.py — HybridRetriever
  - 加载所有分区（FAISS + BM25）
  - 向量检索 + BM25 双路，RRF 融合
  - 元数据 Boost 重排，返回 Top-5
"""

import json
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np


def tokenize_jp(text: str) -> list[str]:
    """与 build_kb.py 保持相同的 bigram 分词"""
    text = re.sub(r'\s+', '', text)
    chars = list(text)
    bigrams = [chars[i] + chars[i+1] for i in range(len(chars) - 1)]
    return chars + bigrams


def rrf_fusion(result_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion
    result_lists: 每路检索返回的 qa_pair index 列表（已按相关度排序）
    """
    scores: dict[int, float] = {}
    for results in result_lists:
        for rank, idx in enumerate(results):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class PartitionIndex:
    """单个分区的索引（FAISS + BM25 + QA数据）"""

    def __init__(self, partition_dir: Path):
        import faiss
        from rank_bm25 import BM25Okapi

        self.name = partition_dir.name
        with open(partition_dir / "qa_pairs.json", 'r', encoding='utf-8') as f:
            self.qa_pairs: list[dict] = json.load(f)

        self.faiss_index = faiss.read_index(str(partition_dir / "faiss.index"))
        with open(partition_dir / "bm25.pkl", 'rb') as f:
            self.bm25: BM25Okapi = pickle.load(f)

    def vector_search(self, query_vec: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        """返回 [(local_index, cosine_score), ...]"""
        scores, indices = self.faiss_index.search(query_vec, top_k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def bm25_search(self, query_tokens: list[str], top_k: int = 20) -> list[tuple[int, float]]:
        """返回 [(local_index, bm25_score), ...]"""
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


class HybridRetriever:
    def __init__(self, kb_dir: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        if kb_dir is None:
            kb_dir = Path(__file__).parent.parent / "kb"
        self.kb_dir = Path(kb_dir)
        parts_dir = self.kb_dir / "partitions"

        if not parts_dir.exists():
            raise FileNotFoundError(
                "分区知识库不存在，请先运行: python scripts/build_kb.py --input <xlsx>"
            )

        self.partitions: dict[str, PartitionIndex] = {}
        for d in parts_dir.iterdir():
            if d.is_dir() and (d / "qa_pairs.json").exists():
                self.partitions[d.name] = PartitionIndex(d)
                print(f"[Retriever] 加载分区 [{d.name}] — {len(self.partitions[d.name].qa_pairs)} 条")

        if not self.partitions:
            raise FileNotFoundError("partitions/ 下未找到有效分区，请重新运行 build_kb.py")

        self.model = SentenceTransformer('intfloat/multilingual-e5-base')
        total = sum(len(p.qa_pairs) for p in self.partitions.values())
        print(f"[Retriever] 全部加载完毕 — {len(self.partitions)} 分区 · {total} 条")

    def search(
        self,
        query: str,
        partition: Optional[str] = None,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> dict:
        """
        返回:
        {
            "partition_used": "不具合",
            "candidates_count": 40,
            "top_results": [
                {
                    "rank": 1,
                    "score": 0.923,        # 最终重排分数（0~1）
                    "vec_score": 0.91,
                    "rrf_score": 0.031,
                    "qa": { ...qa字段... }
                },
                ...
            ]
        }
        """
        # 确定目标分区（找不到则用全库合并）
        target = self.partitions.get(partition) if partition else None

        # 编码查询向量
        query_vec = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True,
        )
        query_vec = np.array(query_vec, dtype=np.float32)
        query_tokens = tokenize_jp(query)

        if target:
            results = self._search_partition(target, query_vec, query_tokens, candidate_k)
            source_name = target.name
            raw_candidate_count = min(candidate_k * 2, len(target.qa_pairs))
        else:
            # 跨分区搜索：每个分区取 candidate_k/2 候选后合并
            merged: list[dict] = []
            for name, part in self.partitions.items():
                partial = self._search_partition(part, query_vec, query_tokens, candidate_k // 2)
                merged.extend(partial)
            # 合并后按 final_score 重排
            merged.sort(key=lambda x: x["final_score"], reverse=True)
            results = merged
            source_name = "全分区"
            raw_candidate_count = sum(len(p.qa_pairs) for p in self.partitions.values())

        # 重排：按 final_score 取 Top-K
        top = results[:top_k]
        for rank, item in enumerate(top, 1):
            item["rank"] = rank

        return {
            "partition_used": source_name,
            "candidates_count": raw_candidate_count,
            "top_results": top,
        }

    def _search_partition(
        self,
        part: PartitionIndex,
        query_vec: np.ndarray,
        query_tokens: list[str],
        candidate_k: int,
    ) -> list[dict]:
        """对单个分区做向量+BM25双路检索 → RRF → 元数据Boost重排"""
        vec_results = part.vector_search(query_vec, top_k=candidate_k)
        bm25_results = part.bm25_search(query_tokens, top_k=candidate_k)

        # 构建分数映射
        vec_score_map = {idx: score for idx, score in vec_results}
        bm25_raw = {idx: score for idx, score in bm25_results}
        # 归一化 BM25（/max）
        bm25_max = max(bm25_raw.values(), default=1.0)
        bm25_norm = {idx: s / bm25_max for idx, s in bm25_raw.items()}

        # RRF
        rrf = rrf_fusion(
            [[i for i, _ in vec_results], [i for i, _ in bm25_results]]
        )
        # 归一化 RRF（/max）
        rrf_max = rrf[0][1] if rrf else 1.0
        rrf_norm = {idx: s / rrf_max for idx, s in rrf[:candidate_k * 2]}

        # 候选集：两路并集
        candidates = set(vec_score_map.keys()) | set(bm25_raw.keys())

        items = []
        for idx in candidates:
            if idx >= len(part.qa_pairs):
                continue
            qa = part.qa_pairs[idx]
            vs = vec_score_map.get(idx, 0.0)
            rs = rrf_norm.get(idx, 0.0)

            # 最终得分：向量权重 70% + RRF 权重 30%
            score = 0.7 * vs + 0.3 * rs
            items.append({
                "rank": 0,          # 重排后填入
                "final_score": round(score, 4),
                "vec_score": round(vs, 4),
                "rrf_score": round(rs, 4),
                "qa": qa,
            })

        items.sort(key=lambda x: x["final_score"], reverse=True)
        return items

    @property
    def top1_confidence(self) -> float:
        """供外部代码快速拿到上次搜索结果的 top-1 分数（已由 search() 返回）"""
        return 0.0  # placeholder，使用 search() 返回值中的 top_results[0]["final_score"]

    @staticmethod
    def detect_turns(email_text: str) -> int:
        quoted = [l for l in email_text.split('\n') if l.startswith('>')]
        return 2 if quoted else 1

    @staticmethod
    def route(confidence: float, turns: int) -> str:
        if turns > 1:
            return "HUMAN"
        if confidence >= 0.90:
            return "AUTO"
        if confidence >= 0.70:
            return "REVIEW"
        return "HUMAN"
```

- [ ] **Step 2: 验证（需要先完成 Task 1 的 Step 6）**

```bash
python -c "
import asyncio
from app.retriever import HybridRetriever
r = HybridRetriever()
result = r.search('ゲームが起動しない', partition='不具合', top_k=5)
print('分区:', result['partition_used'])
print('候选:', result['candidates_count'])
for item in result['top_results']:
    print(f\"  #{item['rank']} score={item['final_score']} | {item['qa']['question_jp'][:50]}\")
"
```

预期：打印 5 条有内容的结果，scores 在 0.5~1.0 之间。

---

## Task 5: 更新 app/llm.py — 新增 translate_email()，更新生成 prompt 使用 Top-5

**Files:**
- Modify: `app/llm.py`

- [ ] **Step 1: 在文件末尾追加 `translate_email()` 和 `translate_reply()` 函数**

```python
async def translate_email(email_jp: str) -> str:
    """
    将日文邮件翻译为中文（非流式，返回完整字符串）。
    用于在 UI 中展示邮件中文版。
    """
    client = get_client()
    resp = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是专业的日中翻译，负责将日文游戏客服邮件翻译为简洁准确的中文。\n"
                    "保留原文段落结构，不添加额外解释，直接输出译文。"
                ),
            },
            {"role": "user", "content": email_jp},
        ],
        temperature=0.1,
        max_tokens=1500,
    )
    return resp.choices[0].message.content.strip()


async def translate_reply(reply_jp: str) -> str:
    """将生成的日文回复翻译为中文，供客服参考"""
    client = get_client()
    resp = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "将以下日文客服回复翻译为中文，保持正式礼貌语气，直接输出译文。",
            },
            {"role": "user", "content": reply_jp},
        ],
        temperature=0.1,
        max_tokens=1000,
    )
    return resp.choices[0].message.content.strip()
```

- [ ] **Step 2: 更新三个生成函数，使用 top_results（list[dict]）替代 similar_answer_jp**

将 `generate_auto_reply` 的签名改为：

```python
async def generate_auto_reply(
    question_jp: str,
    top_results: list[dict],   # 来自 HybridRetriever.search() 的 top_results
) -> AsyncGenerator[str, None]:
    # 构建参考上下文（最多用 Top-3，避免 token 过长）
    refs = ""
    for item in top_results[:3]:
        qa = item["qa"]
        refs += f"\n---\n問題: {qa['question_jp'][:200]}\n回答: {qa['answer_jp'][:400]}"

    messages = [
        {
            "role": "system",
            "content": (
                f"你是游戏《{GAME}》的客服专员，负责用日语回复玩家的问题。\n"
                "请参考以下历史参考案例（按相关度排序），针对玩家当前问题生成一封简洁、礼貌的日文回复邮件。\n"
                "要求：1. 开头「いつもご利用いただきありがとうございます。」\n"
                "2. 针对本次问题，不要照抄 3. 结尾「今後ともよろしくお願いいたします。」\n"
                "4. 只输出邮件正文"
            )
        },
        {
            "role": "user",
            "content": f"【玩家问题】\n{question_jp}\n\n【参考案例（Top-3）】{refs}"
        }
    ]
    async for token in _stream(messages):
        yield token
```

将 `generate_review_draft` 签名改为：

```python
async def generate_review_draft(
    question_jp: str,
    top_results: list[dict],
) -> AsyncGenerator[str, None]:
    refs = ""
    for item in top_results[:3]:
        qa = item["qa"]
        refs += (
            f"\n---\n問題JP: {qa['question_jp'][:150]}\n"
            f"问题ZH: {qa['question_zh'][:150]}\n"
            f"回答JP: {qa['answer_jp'][:300]}\n"
            f"回答ZH: {qa['answer_zh'][:300]}"
        )
    messages = [
        {
            "role": "system",
            "content": (
                f"你是游戏《{GAME}》的AI助手，辅助中国客服处理日文玩家邮件。\n"
                "请完成两件事，用分隔符 [SUMMARY_END] 隔开：\n"
                "第一部分 — 中文摘要（3-5句）：玩家反馈了什么、情绪、核心诉求\n"
                "[SUMMARY_END]\n"
                "第二部分 — 日文回复草稿：参考历史案例语气，结构完整\n"
                "只输出这两部分，无其他说明"
            )
        },
        {
            "role": "user",
            "content": f"【玩家原文（日文）】\n{question_jp}\n\n【参考案例（Top-3）】{refs}"
        }
    ]
    async for token in _stream(messages):
        yield token
```

将 `generate_human_summary` 签名保持不变（HUMAN 档不需要参考案例）。

---

## Task 6: 更新 app/main.py — 新 SSE 协议 + 分析端点

**Files:**
- Modify: `app/main.py`

### 新 SSE 事件协议
```
event: translate   data: {"email_zh": "中文译文..."}        ← 邮件翻译完成
event: intent      data: {"partition": "不具合"}             ← 意图分类结果
event: meta        data: {"confidence": 87.2, "level": "REVIEW", "partition": "不具合",
                          "candidates_count": 40, "top_results": [...5条...]}
event: section     data: {"name": "summary"|"reply"}
event: token       data: {"text": "...", "section": "summary"|"reply"}
event: reply_zh    data: {"text": "完整中文译文"}             ← 回复翻译完成（非流式）
event: done        data: {}
event: error       data: {"message": "..."}
```

- [ ] **Step 1: 完整替换 main.py**

```python
"""
main.py — FastAPI 后端 v2
  GET  /                → 前端页面
  POST /api/process     → SSE 流式处理邮件（新协议）
  GET  /api/analytics   → 返回 analytics.json
  GET  /api/health      → 健康检查
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from app.classifier import classify_intent
from app.llm import (
    generate_auto_reply,
    generate_human_summary,
    generate_review_draft,
    translate_email,
    translate_reply,
)
from app.retriever import HybridRetriever

app = FastAPI(title="ぽちゃガチョ！客服 AI v2")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_retriever: HybridRetriever | None = None
KB_DIR = Path(__file__).parent.parent / "kb"


@app.on_event("startup")
async def startup():
    global _retriever
    try:
        _retriever = HybridRetriever()
    except FileNotFoundError as e:
        print(f"[警告] {e}")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (static_dir / "index.html").read_text(encoding='utf-8')


@app.get("/api/health")
async def health():
    kb_ready = _retriever is not None
    total = sum(len(p.qa_pairs) for p in _retriever.partitions.values()) if kb_ready else 0
    partitions = list(_retriever.partitions.keys()) if kb_ready else []
    return {"status": "ok", "kb_ready": kb_ready, "total": total, "partitions": partitions}


@app.get("/api/analytics")
async def analytics():
    path = KB_DIR / "analytics.json"
    if not path.exists():
        raise HTTPException(404, "analytics.json 不存在，请运行 scripts/analyze_kb.py")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


class ProcessRequest(BaseModel):
    email_text: str


@app.post("/api/process")
async def process_email(req: ProcessRequest):
    email_text = req.email_text.strip()
    if not email_text:
        raise HTTPException(400, "邮件内容不能为空")
    if _retriever is None:
        raise HTTPException(503, "知识库未加载")

    async def stream():
        try:
            # 1. 并行：邮件翻译 + 意图分类
            email_zh, partition = await asyncio.gather(
                translate_email(email_text),
                classify_intent(email_text),
            )
            yield f"event: translate\ndata: {json.dumps({'email_zh': email_zh}, ensure_ascii=False)}\n\n"
            yield f"event: intent\ndata: {json.dumps({'partition': partition}, ensure_ascii=False)}\n\n"

            # 2. 混合检索 + 重排 Top-5
            turns = HybridRetriever.detect_turns(email_text)
            search_result = _retriever.search(email_text, partition=partition, top_k=5)
            top_results = search_result["top_results"]

            confidence = top_results[0]["final_score"] if top_results else 0.0
            level = HybridRetriever.route(confidence, turns)

            # 序列化 top_results（去掉大字段，只保留展示需要的）
            top_for_frontend = [
                {
                    "rank": r["rank"],
                    "score": r["final_score"],
                    "vec_score": r["vec_score"],
                    "inquiry_type": r["qa"].get("inquiry_type", ""),
                    "environment": r["qa"].get("environment", ""),
                    "app_version": r["qa"].get("app_version", ""),
                    "date": r["qa"].get("date", ""),
                    "question_jp": r["qa"].get("question_jp", ""),
                    "question_zh": r["qa"].get("question_zh", ""),
                    "answer_jp": r["qa"].get("answer_jp", ""),
                    "answer_zh": r["qa"].get("answer_zh", ""),
                }
                for r in top_results
            ]

            meta = {
                "confidence": round(confidence * 100, 1),
                "level": level,
                "turns": turns,
                "partition": search_result["partition_used"],
                "candidates_count": search_result["candidates_count"],
                "top_results": top_for_frontend,
            }
            yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

            # 3. LLM 生成
            reply_jp_parts = []

            if level == "AUTO":
                yield f"event: section\ndata: {json.dumps({'name': 'reply'})}\n\n"
                async for token in generate_auto_reply(email_text, top_results):
                    reply_jp_parts.append(token)
                    yield f"event: token\ndata: {json.dumps({'text': token, 'section': 'reply'}, ensure_ascii=False)}\n\n"

            elif level == "REVIEW":
                yield f"event: section\ndata: {json.dumps({'name': 'summary'})}\n\n"
                summary_done = False
                async for token in generate_review_draft(email_text, top_results):
                    if "[SUMMARY_END]" in token and not summary_done:
                        summary_done = True
                        yield f"event: section\ndata: {json.dumps({'name': 'reply'})}\n\n"
                        after = token.split("[SUMMARY_END]", 1)[-1]
                        if after.strip():
                            reply_jp_parts.append(after)
                            yield f"event: token\ndata: {json.dumps({'text': after, 'section': 'reply'}, ensure_ascii=False)}\n\n"
                    else:
                        section = "reply" if summary_done else "summary"
                        if summary_done:
                            reply_jp_parts.append(token)
                        yield f"event: token\ndata: {json.dumps({'text': token, 'section': section}, ensure_ascii=False)}\n\n"

            else:  # HUMAN
                yield f"event: section\ndata: {json.dumps({'name': 'summary'})}\n\n"
                async for token in generate_human_summary(email_text):
                    yield f"event: token\ndata: {json.dumps({'text': token, 'section': 'summary'}, ensure_ascii=False)}\n\n"

            # 4. 翻译日文回复为中文
            if reply_jp_parts:
                reply_jp_full = "".join(reply_jp_parts)
                reply_zh = await translate_reply(reply_jp_full)
                yield f"event: reply_zh\ndata: {json.dumps({'text': reply_zh}, ensure_ascii=False)}\n\n"

            yield f"event: done\ndata: {{}}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

---

## Task 7: 重写 app/static/index.html — 新双栏布局 + Top-5 来源卡片 + 数据看板

**Files:**
- Rewrite: `app/static/index.html`

### 布局结构
```
Header (知识库状态 · 分区列表 · Tab切换)
────────────────────────────────────────
[处理邮件 Tab]
  ┌──────────────────┬──────────────────┐
  │ 📥 邮件输入       │ 📤 生成回复       │
  │ [textarea]       │ 置信度+等级badge  │
  │ [处理] btn       │ ─────────────── │
  │ ─────────────── │ 🇯🇵 日文回复 [复制]│
  │ 🇨🇳 邮件中文译文  │ 🇨🇳 回复中文译文  │
  └──────────────────┴──────────────────┘
  ┌── 检索来源 ──────────────────────────┐
  │ 分区: 不具合 | 候选: 40条 | Top-5重排│
  │ [#1 0.94 | 版本 | 机型 | 日期]      │
  │ [JP问题] [ZH问题] [JP回复] [ZH回复]  │
  │ #2 ... #3 ... (可折叠)              │
  └─────────────────────────────────────┘

[数据看板 Tab]
  ┌────────────┬────────────┐
  │ 问询类型分布│ 版本问题热点│
  └────────────┴────────────┘
  ┌────────────┬────────────┐
  │ 时间趋势    │ 设备分布   │
  └────────────┴────────────┘
```

- [ ] **Step 1: 完整替换 index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ぽちゃガチョ！客服 AI v2</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a; --border: #2d3148;
      --text: #e2e8f0; --muted: #718096; --accent: #6c63ff;
      --auto: #38a169; --review: #d69e2e; --human: #e53e3e;
      --r: 10px;
    }
    body { background: var(--bg); color: var(--text);
           font-family: 'Segoe UI','Noto Sans SC','Noto Sans JP',system-ui,sans-serif;
           min-height: 100vh; display: flex; flex-direction: column; }

    /* ── Header ── */
    header { padding: 14px 28px; border-bottom: 1px solid var(--border);
             display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    header h1 { font-size: 15px; font-weight: 600; }
    .kb-status { font-size: 12px; color: var(--muted); margin-left: auto;
                 display: flex; align-items: center; gap: 6px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #555; }
    .dot.ok { background: var(--auto); }

    /* ── Tabs ── */
    .tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); padding: 0 28px; }
    .tab-btn { padding: 10px 20px; font-size: 13px; cursor: pointer;
               border: none; background: none; color: var(--muted);
               border-bottom: 2px solid transparent; transition: color 0.15s; }
    .tab-btn.active { color: var(--text); border-bottom-color: var(--accent); }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }

    /* ── Main grid ── */
    .process-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
                    padding: 20px 28px; }
    @media (max-width: 900px) { .process-grid { grid-template-columns: 1fr; } }

    /* ── Panels ── */
    .panel { background: var(--surface); border: 1px solid var(--border);
             border-radius: var(--r); padding: 18px; display: flex;
             flex-direction: column; gap: 12px; }
    .panel-title { font-size: 12px; font-weight: 600; color: var(--muted);
                   text-transform: uppercase; letter-spacing: 0.06em; }

    textarea { width: 100%; height: 220px; background: var(--surface2);
               border: 1px solid var(--border); border-radius: 8px;
               color: var(--text); font-size: 13px; line-height: 1.7;
               padding: 10px 12px; resize: vertical; outline: none; font-family: inherit; }
    textarea:focus { border-color: var(--accent); }

    .btn-primary { background: var(--accent); color: #fff; border: none;
                   border-radius: 8px; padding: 9px 20px; font-size: 13px;
                   font-weight: 600; cursor: pointer; align-self: flex-start; }
    .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-copy { background: #2d3148; color: var(--text); border: none;
                border-radius: 6px; padding: 4px 10px; font-size: 11px; cursor: pointer; }
    .btn-copy:hover { background: #3a3f5c; }

    /* ── Translation box ── */
    .translation-box { background: var(--surface2); border-radius: 8px;
                        padding: 12px; font-size: 13px; line-height: 1.8;
                        color: #a0aec0; white-space: pre-wrap; min-height: 60px;
                        max-height: 180px; overflow-y: auto; }

    /* ── Confidence badge ── */
    .conf-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .conf-score { font-size: 28px; font-weight: 700; }
    .level-badge { padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
    .level-AUTO   { background: #276749; color: #68d391; }
    .level-REVIEW { background: #744210; color: #f6e05e; }
    .level-HUMAN  { background: #742a2a; color: #fc8181; }
    .part-tag { background: var(--surface2); color: var(--muted); padding: 3px 8px;
                border-radius: 4px; font-size: 11px; }

    /* ── Reply section ── */
    .reply-header { display: flex; align-items: center; gap: 8px; }
    .reply-label { font-size: 12px; color: var(--muted); flex: 1; }
    .reply-text { background: var(--surface2); border-radius: 8px; padding: 12px;
                  font-size: 13px; line-height: 1.8; white-space: pre-wrap;
                  min-height: 60px; max-height: 240px; overflow-y: auto; }
    .reply-text.streaming { border: 1px solid var(--accent); }
    .cursor::after { content: '▋'; animation: blink 0.8s step-end infinite; color: var(--accent); }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

    /* ── Retrieval panel ── */
    .retrieval-panel { margin: 0 28px 20px; background: var(--surface);
                       border: 1px solid var(--border); border-radius: var(--r); }
    .retrieval-header { display: flex; align-items: center; gap: 12px;
                        padding: 14px 18px; cursor: pointer; user-select: none; }
    .retrieval-header h3 { font-size: 13px; font-weight: 600; }
    .ret-stat { background: var(--surface2); padding: 3px 8px; border-radius: 4px;
                font-size: 11px; color: var(--muted); }
    .expand-icon { margin-left: auto; color: var(--muted); font-size: 12px; }
    .retrieval-body { display: none; padding: 0 18px 16px; }
    .retrieval-body.open { display: block; }

    /* ── Top-5 result cards ── */
    .result-card { border: 1px solid var(--border); border-radius: 8px;
                   margin-bottom: 10px; overflow: hidden; }
    .result-card-header { display: flex; align-items: center; gap: 8px;
                          padding: 10px 14px; background: var(--surface2); flex-wrap: wrap; }
    .rank-badge { background: var(--accent); color: #fff; width: 22px; height: 22px;
                  border-radius: 50%; display: flex; align-items: center; justify-content: center;
                  font-size: 11px; font-weight: 700; flex-shrink: 0; }
    .score-bar { flex: 1; max-width: 120px; }
    .score-bar-fill { height: 6px; border-radius: 3px; background: var(--accent); }
    .meta-tags { display: flex; gap: 6px; flex-wrap: wrap; }
    .meta-tag { background: #2d3148; color: var(--muted); padding: 2px 7px;
                border-radius: 4px; font-size: 10px; }
    .result-card-body { display: grid; grid-template-columns: 1fr 1fr;
                        gap: 0; border-top: 1px solid var(--border); }
    @media (max-width: 700px) { .result-card-body { grid-template-columns: 1fr; } }
    .result-col { padding: 10px 14px; }
    .result-col + .result-col { border-left: 1px solid var(--border); }
    .result-col-title { font-size: 10px; color: var(--muted); margin-bottom: 4px; }
    .result-jp { font-size: 12px; line-height: 1.7; max-height: 80px; overflow-y: auto; }
    .result-zh { font-size: 12px; color: #a0aec0; line-height: 1.7; max-height: 80px; overflow-y: auto; }

    /* ── Dashboard ── */
    .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr;
                      gap: 20px; padding: 20px 28px; }
    @media (max-width: 800px) { .dashboard-grid { grid-template-columns: 1fr; } }
    .chart-card { background: var(--surface); border: 1px solid var(--border);
                  border-radius: var(--r); padding: 18px; }
    .chart-title { font-size: 12px; color: var(--muted); font-weight: 600;
                   text-transform: uppercase; margin-bottom: 14px; }
    canvas { max-height: 220px; }

    /* ── Status & Error ── */
    .status-bar { font-size: 12px; color: var(--muted); min-height: 18px; }
    .spinner { display: inline-block; width: 12px; height: 12px;
               border: 2px solid #444; border-top-color: var(--accent);
               border-radius: 50%; animation: spin 0.7s linear infinite;
               vertical-align: middle; margin-right: 4px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .error-box { display: none; background: #2d1b1b; border: 1px solid var(--human);
                 border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #fc8181; }
    .error-box.show { display: block; }
  </style>
</head>
<body>

<header>
  <span style="font-size:20px">🎮</span>
  <h1>ぽちゃガチョ！客服 AI <span style="color:var(--muted);font-weight:400">v2</span></h1>
  <div class="kb-status">
    <span class="dot" id="kb-dot"></span>
    <span id="kb-info">加载中...</span>
  </div>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('process', this)">📧 处理邮件</button>
  <button class="tab-btn" onclick="switchTab('dashboard', this)">📊 数据看板</button>
</div>

<!-- ══ 处理邮件 Tab ══ -->
<div id="tab-process" class="tab-pane active">

  <div class="process-grid">
    <!-- 左栏：输入 + 邮件译文 -->
    <div class="panel">
      <div class="panel-title">📥 粘贴日文玩家邮件</div>
      <textarea id="email-input" placeholder="将玩家发来的日文邮件正文粘贴到这里...（Ctrl+Enter 提交）"></textarea>
      <div style="display:flex;gap:8px;align-items:center">
        <button class="btn-primary" id="process-btn" onclick="processEmail()">处理邮件</button>
        <div class="status-bar" id="status-bar"></div>
      </div>
      <div class="error-box" id="error-box"></div>

      <!-- 邮件中文译文 -->
      <div id="email-zh-block" style="display:none;flex-direction:column;gap:6px">
        <div class="panel-title">🇨🇳 邮件中文译文</div>
        <div class="translation-box" id="email-zh-text"></div>
      </div>
    </div>

    <!-- 右栏：生成回复 -->
    <div class="panel">
      <div class="panel-title">📤 AI 生成回复</div>

      <!-- 置信度卡片 -->
      <div id="conf-card" style="display:none">
        <div class="conf-row">
          <div class="conf-score" id="conf-score" style="color:var(--auto)">—</div>
          <div>
            <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">
              <span class="level-badge" id="level-badge">—</span>
              <span class="part-tag" id="part-tag"></span>
            </div>
            <div style="font-size:11px;color:var(--muted);margin-top:4px" id="conf-desc"></div>
          </div>
        </div>
      </div>

      <!-- 中文摘要（REVIEW/HUMAN档） -->
      <div id="summary-block" style="display:none;flex-direction:column;gap:6px">
        <div class="panel-title">🇨🇳 中文分析摘要</div>
        <div class="reply-text" id="summary-text"></div>
      </div>

      <!-- 日文回复 -->
      <div id="reply-block" style="display:none;flex-direction:column;gap:6px">
        <div class="reply-header">
          <div class="reply-label" id="reply-label">🇯🇵 日文回复</div>
          <button class="btn-copy" id="copy-btn" onclick="copyReply()" style="display:none">复制</button>
        </div>
        <div class="reply-text" id="reply-text"></div>
      </div>

      <!-- 中文回复译文 -->
      <div id="reply-zh-block" style="display:none;flex-direction:column;gap:6px">
        <div class="panel-title">🇨🇳 回复中文译文（参考）</div>
        <div class="translation-box" id="reply-zh-text"></div>
      </div>
    </div>
  </div>

  <!-- 检索来源面板 -->
  <div class="retrieval-panel" id="retrieval-panel" style="display:none">
    <div class="retrieval-header" onclick="toggleRetrieval()">
      <h3>📚 检索来源</h3>
      <span class="ret-stat" id="ret-partition"></span>
      <span class="ret-stat" id="ret-count"></span>
      <span class="ret-stat" id="ret-top"></span>
      <span class="expand-icon" id="expand-icon">▼ 展开</span>
    </div>
    <div class="retrieval-body" id="retrieval-body"></div>
  </div>

</div>

<!-- ══ 数据看板 Tab ══ -->
<div id="tab-dashboard" class="tab-pane">
  <div class="dashboard-grid">
    <div class="chart-card">
      <div class="chart-title">问询类型分布</div>
      <canvas id="chart-inquiry"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">版本问题热点（Top 10 版本）</div>
      <canvas id="chart-version"></canvas>
    </div>
    <div class="chart-card" style="grid-column: span 2">
      <div class="chart-title">月度问询趋势</div>
      <canvas id="chart-monthly" style="max-height:180px"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">设备/平台分布</div>
      <canvas id="chart-env"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title">知识库分区分布</div>
      <canvas id="chart-partition"></canvas>
    </div>
  </div>
</div>

<script>
// ── Tab 切换 ──
function switchTab(name, btn) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'dashboard') loadDashboard();
}

// ── Health check ──
async function checkHealth() {
  try {
    const d = await fetch('/api/health').then(r => r.json());
    const dot = document.getElementById('kb-dot');
    const info = document.getElementById('kb-info');
    if (d.kb_ready) {
      dot.classList.add('ok');
      info.textContent = `知识库就绪 · ${d.total} 条 · 分区: ${d.partitions.join(' / ')}`;
    } else {
      info.textContent = '知识库未加载，请运行 build_kb.py';
    }
  } catch(e) { document.getElementById('kb-info').textContent = '服务连接失败'; }
}
checkHealth();

// ── Utilities ──
function setStatus(msg) {
  document.getElementById('status-bar').innerHTML = msg
    ? `<span class="spinner"></span>${msg}` : '';
}
function showError(msg) {
  const el = document.getElementById('error-box');
  el.textContent = msg; el.classList.add('show');
}
function hideAll() {
  document.getElementById('error-box').classList.remove('show');
  ['conf-card','summary-block','reply-block','reply-zh-block','email-zh-block']
    .forEach(id => document.getElementById(id).style.display = 'none');
  document.getElementById('retrieval-panel').style.display = 'none';
  ['summary-text','reply-text','reply-zh-text','email-zh-text']
    .forEach(id => { const el = document.getElementById(id);
                     el.textContent = ''; el.classList.remove('streaming','cursor'); });
  document.getElementById('copy-btn').style.display = 'none';
}

const LEVEL_COLOR = {AUTO:'#38a169', REVIEW:'#d69e2e', HUMAN:'#e53e3e'};
const LEVEL_DESC  = {
  AUTO:   '置信度高，可自动发出',
  REVIEW: '置信度中等，请确认后发出',
  HUMAN:  '置信度低或多轮，建议人工回复',
};

function renderMeta(meta) {
  document.getElementById('conf-card').style.display = 'block';
  const score = document.getElementById('conf-score');
  score.textContent = meta.confidence + '%';
  score.style.color = LEVEL_COLOR[meta.level] || '#ccc';
  const badge = document.getElementById('level-badge');
  badge.textContent = meta.level; badge.className = 'level-badge level-' + meta.level;
  document.getElementById('part-tag').textContent = '分区: ' + meta.partition;
  document.getElementById('conf-desc').textContent = LEVEL_DESC[meta.level] || '';

  // 回复标签
  const rl = document.getElementById('reply-label');
  rl.textContent = meta.level === 'AUTO' ? '🇯🇵 日文回复（自动档）' : '🇯🇵 日文回复草稿';

  // 复制按钮仅 AUTO 档显示
  if (meta.level === 'AUTO') {
    document.getElementById('copy-btn').style.display = 'inline-block';
  }

  // 检索来源面板
  if (meta.top_results && meta.top_results.length > 0) {
    renderRetrieval(meta);
  }
}

function renderRetrieval(meta) {
  const panel = document.getElementById('retrieval-panel');
  panel.style.display = 'block';
  document.getElementById('ret-partition').textContent = '分区: ' + meta.partition;
  document.getElementById('ret-count').textContent = '候选: ' + meta.candidates_count + ' 条';
  document.getElementById('ret-top').textContent = '重排 Top-' + meta.top_results.length;

  const body = document.getElementById('retrieval-body');
  body.innerHTML = '';

  meta.top_results.forEach(r => {
    const pct = Math.round(r.score * 100);
    const card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML = `
      <div class="result-card-header">
        <div class="rank-badge">${r.rank}</div>
        <div class="score-bar" title="${pct}%">
          <div class="score-bar-fill" style="width:${pct}%"></div>
        </div>
        <span style="font-size:12px;font-weight:600;color:${pct>=80?'#68d391':pct>=60?'#f6e05e':'#fc8181'}">${pct}%</span>
        <div class="meta-tags">
          ${r.inquiry_type ? `<span class="meta-tag">📋 ${r.inquiry_type}</span>` : ''}
          ${r.environment  ? `<span class="meta-tag">📱 ${r.environment}</span>` : ''}
          ${r.app_version  ? `<span class="meta-tag">🔖 v${r.app_version}</span>` : ''}
          ${r.date         ? `<span class="meta-tag">📅 ${r.date}</span>` : ''}
        </div>
      </div>
      <div class="result-card-body">
        <div class="result-col">
          <div class="result-col-title">🇯🇵 原始提问</div>
          <div class="result-jp">${escHtml(r.question_jp)}</div>
          <div style="margin-top:8px" class="result-col-title">🇨🇳 提问译文</div>
          <div class="result-zh">${escHtml(r.question_zh)}</div>
        </div>
        <div class="result-col">
          <div class="result-col-title">🇯🇵 知识库回复</div>
          <div class="result-jp">${escHtml(r.answer_jp)}</div>
          <div style="margin-top:8px" class="result-col-title">🇨🇳 回复译文</div>
          <div class="result-zh">${escHtml(r.answer_zh)}</div>
        </div>
      </div>`;
    body.appendChild(card);
  });
}

function toggleRetrieval() {
  const body = document.getElementById('retrieval-body');
  const icon = document.getElementById('expand-icon');
  const open = body.classList.toggle('open');
  icon.textContent = open ? '▲ 收起' : '▼ 展开';
}

function escHtml(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function copyReply() {
  const text = document.getElementById('reply-text').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.textContent = '✓ 已复制'; setTimeout(() => btn.textContent = '复制', 2000);
  });
}

// ── 主处理流程 ──
async function processEmail() {
  const emailText = document.getElementById('email-input').value.trim();
  if (!emailText) { showError('请先粘贴邮件内容'); return; }

  hideAll();
  setStatus('翻译邮件 + 意图分类...');
  document.getElementById('process-btn').disabled = true;

  const resp = await fetch('/api/process', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({email_text: emailText}),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({detail: resp.statusText}));
    showError(err.detail || '请求失败');
    setStatus(''); document.getElementById('process-btn').disabled = false; return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = '', currentSection = null;

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream: true});
    const lines = buf.split('\n'); buf = lines.pop();
    let evType = null;
    for (const line of lines) {
      if (line.startsWith('event: ')) { evType = line.slice(7).trim(); continue; }
      if (!line.startsWith('data: ')) continue;
      let data; try { data = JSON.parse(line.slice(6)); } catch { continue; }

      if (evType === 'translate') {
        const block = document.getElementById('email-zh-block');
        block.style.display = 'flex';
        document.getElementById('email-zh-text').textContent = data.email_zh || '';
        setStatus('意图分类...');

      } else if (evType === 'intent') {
        setStatus('向量+BM25 混合检索...');

      } else if (evType === 'meta') {
        setStatus('AI 生成回复...');
        renderMeta(data);

      } else if (evType === 'section') {
        currentSection = data.name;
        if (currentSection === 'summary') {
          const b = document.getElementById('summary-block');
          b.style.display = 'flex';
          document.getElementById('summary-text').classList.add('streaming', 'cursor');
        } else if (currentSection === 'reply') {
          document.getElementById('summary-text').classList.remove('streaming','cursor');
          const b = document.getElementById('reply-block');
          b.style.display = 'flex';
          document.getElementById('reply-text').classList.add('streaming', 'cursor');
        }

      } else if (evType === 'token') {
        const el = document.getElementById(data.section === 'reply' ? 'reply-text' : 'summary-text');
        el.textContent += data.text;

      } else if (evType === 'reply_zh') {
        document.getElementById('reply-text').classList.remove('streaming','cursor');
        const b = document.getElementById('reply-zh-block');
        b.style.display = 'flex';
        document.getElementById('reply-zh-text').textContent = data.text || '';

      } else if (evType === 'done') {
        document.getElementById('summary-text').classList.remove('streaming','cursor');
        document.getElementById('reply-text').classList.remove('streaming','cursor');
        setStatus('');

      } else if (evType === 'error') {
        const msg = data.message || '处理出错';
        if (msg.includes('402') || msg.includes('Insufficient Balance')) {
          showError('⚠️ DeepSeek 余额不足（402），请前往 platform.deepseek.com 充值。');
        } else { showError(msg); }
        setStatus('');
      }
      evType = null;
    }
  }
  document.getElementById('process-btn').disabled = false;
}

document.getElementById('email-input').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') processEmail();
});

// ── 数据看板 ──
let chartsBuilt = false;
async function loadDashboard() {
  if (chartsBuilt) return;
  try {
    const data = await fetch('/api/analytics').then(r => r.json());
    buildCharts(data);
    chartsBuilt = true;
  } catch(e) {
    console.error('看板数据加载失败:', e);
  }
}

const CHART_COLORS = [
  '#6c63ff','#38a169','#d69e2e','#e53e3e','#3182ce',
  '#805ad5','#dd6b20','#319795','#e53e3e','#718096',
];

function buildCharts(d) {
  const dark = {
    color: '#e2e8f0',
    plugins: { legend: { labels: { color: '#a0aec0', font: {size:11} } } },
    scales: {
      x: { ticks: { color: '#718096', font:{size:10} }, grid: { color: '#2d3148' } },
      y: { ticks: { color: '#718096', font:{size:10} }, grid: { color: '#2d3148' } },
    }
  };

  // 问询类型（Doughnut）
  const inqKeys = Object.keys(d.inquiry_type_dist || {}).slice(0, 8);
  new Chart(document.getElementById('chart-inquiry'), {
    type: 'doughnut',
    data: { labels: inqKeys, datasets: [{
      data: inqKeys.map(k => d.inquiry_type_dist[k]),
      backgroundColor: CHART_COLORS,
    }]},
    options: { ...dark, scales: undefined },
  });

  // 版本热点（横向 Bar）
  const verKeys = Object.keys(d.version_dist || {}).slice(0, 10);
  new Chart(document.getElementById('chart-version'), {
    type: 'bar',
    data: { labels: verKeys, datasets: [{
      label: '问题数',
      data: verKeys.map(k => d.version_dist[k]),
      backgroundColor: '#6c63ff88',
      borderColor: '#6c63ff',
      borderWidth: 1,
    }]},
    options: { ...dark, indexAxis: 'y' },
  });

  // 月度趋势（Line）
  const months = Object.keys(d.date_monthly || {}).filter(k => k !== '不明');
  new Chart(document.getElementById('chart-monthly'), {
    type: 'line',
    data: { labels: months, datasets: [{
      label: '月度问询量',
      data: months.map(k => d.date_monthly[k]),
      borderColor: '#6c63ff',
      backgroundColor: '#6c63ff22',
      fill: true,
      tension: 0.3,
      pointRadius: 3,
    }]},
    options: dark,
  });

  // 设备分布（Bar）
  const envKeys = Object.keys(d.environment_dist || {}).slice(0, 8);
  new Chart(document.getElementById('chart-env'), {
    type: 'bar',
    data: { labels: envKeys, datasets: [{
      label: '数量',
      data: envKeys.map(k => d.environment_dist[k]),
      backgroundColor: CHART_COLORS,
    }]},
    options: dark,
  });

  // 分区分布（Doughnut）
  const partKeys = Object.keys(d.by_partition || {});
  new Chart(document.getElementById('chart-partition'), {
    type: 'doughnut',
    data: { labels: partKeys, datasets: [{
      data: partKeys.map(k => d.by_partition[k]),
      backgroundColor: CHART_COLORS,
    }]},
    options: { ...dark, scales: undefined },
  });
}
</script>

</body>
</html>
```

- [ ] **Step 2: 验证服务运行**

```bash
uvicorn app.main:app --reload --port 8000
```

访问 `http://localhost:8000`：
- Header 显示分区列表（不具合 / 意見要望 / 購入 / その他）
- 粘贴日文邮件 → 左下显示中文译文 → 右侧显示置信度 + 日文回复 + 中文译文
- 底部检索面板展开可见 5 张结果卡片（含 JP/ZH 双栏 + 元数据标签）
- 切换「数据看板」Tab → 5 张 Chart.js 图表

---

## 运行完整流程（总结）

```bash
# 1. 安装新依赖
pip install rank_bm25

# 2. 重建分区知识库（翻译有缓存，速度很快）
python scripts/build_kb.py --input D:/agenticRAG/RAW-emailQA-DATA.xlsx

# 3. 生成分析数据
python scripts/analyze_kb.py

# 4. 启动服务
uvicorn app.main:app --reload --port 8000
```

---

## 自审：Spec 覆盖检查

| 需求 | 对应 Task |
|------|-----------|
| 按问询种类分区 | Task 1 (PARTITION_MAP) |
| 按机型/版本/时间存储元数据 | Task 1 (date/app_version 提取) |
| 意图识别路由至对应 KB | Task 3 (classifier.py) + Task 6 (main.py 并行分类) |
| 向量+BM25 混合检索 | Task 4 (HybridRetriever) |
| RRF 融合 | Task 4 (rrf_fusion) |
| 重排序 Top-5 | Task 4 (final_score 排序) |
| 页面展示 Top-5 全部 | Task 7 (result-card × 5) |
| 邮件下方显示中文译文 | Task 7 (email-zh-block) |
| 右侧日文回复+中文译文 | Task 6 (reply_zh event) + Task 7 |
| 高置信度一键复制 | Task 7 (copy-btn, AUTO 档显示) |
| 低置信度仅推荐思路 | Task 7 (HUMAN: 只显示 summary, 无 reply) |
| 来源 KB 标注 | Task 7 (ret-partition badge) |
| QA 对日中对照 | Task 7 (result-card-body 双栏) |
| 元数据标签（版本/机型/日期） | Task 7 (meta-tags) |
| 数据看板 | Task 2 (analytics.json) + Task 7 (Chart.js) |
| 问询类型分布图 | Task 7 (chart-inquiry Doughnut) |
| 版本问题热点 | Task 7 (chart-version Bar) |
| 时间趋势 | Task 7 (chart-monthly Line) |
| 设备分布 | Task 7 (chart-env Bar) |
