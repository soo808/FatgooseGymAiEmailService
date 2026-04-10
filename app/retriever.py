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
                    "final_score": 0.923,
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
            merged.sort(key=lambda x: x["final_score"], reverse=True)
            results = merged
            source_name = "全分区"
            raw_candidate_count = sum(len(p.qa_pairs) for p in self.partitions.values())

        # 取 Top-K，补充 rank 编号
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
        """对单个分区做向量+BM25双路检索 → RRF → 重排"""
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
                "rank": 0,
                "final_score": round(score, 4),
                "vec_score": round(vs, 4),
                "rrf_score": round(rs, 4),
                "qa": qa,
            })

        items.sort(key=lambda x: x["final_score"], reverse=True)
        return items

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
