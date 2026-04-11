"""
retriever.py — HybridRetriever
  - 加载所有分区（FAISS + BM25）
  - 向量检索 + BM25 双路，RRF 融合
  - per-partition 配置驱动权重 / 阈值
  - reload_partition() 热更新，reload_config() 热更新配置
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from app.indexer import get_model, tokenize_jp


def rrf_fusion(result_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
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
        scores, indices = self.faiss_index.search(query_vec, top_k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]

    def bm25_search(self, query_tokens: list[str], top_k: int = 20) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


class HybridRetriever:
    def __init__(self, kb_dir: Optional[str] = None):
        if kb_dir is None:
            kb_dir = Path(__file__).parent.parent / "kb"
        self.kb_dir = Path(kb_dir)
        parts_dir = self.kb_dir / "partitions"

        if not parts_dir.exists():
            raise FileNotFoundError(
                "分区知识库不存在，请先运行: python scripts/build_kb.py --input <xlsx>"
            )

        self.config = self._load_config()

        self.partitions: dict[str, PartitionIndex] = {}
        for d in parts_dir.iterdir():
            if d.is_dir() and (d / "qa_pairs.json").exists():
                self.partitions[d.name] = PartitionIndex(d)
                print(f"[Retriever] 加载分区 [{d.name}] — {len(self.partitions[d.name].qa_pairs)} 条")

        if not self.partitions:
            raise FileNotFoundError("partitions/ 下未找到有效分区，请重新运行 build_kb.py")

        self.model = get_model()
        total = sum(len(p.qa_pairs) for p in self.partitions.values())
        print(f"[Retriever] 全部加载完毕 — {len(self.partitions)} 分区 · {total} 条")

    def _load_config(self) -> dict:
        cfg_path = self.kb_dir / "retrieval_config.json"
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def reload_config(self) -> None:
        """PUT /api/kb/config 后调用，热更新权重/阈值"""
        self.config = self._load_config()

    def reload_partition(self, name: str) -> None:
        """重建索引完成后调用，热加载指定分区"""
        part_dir = self.kb_dir / "partitions" / name
        if part_dir.exists() and (part_dir / "qa_pairs.json").exists():
            self.partitions[name] = PartitionIndex(part_dir)
            print(f"[Retriever] 热更新分区 [{name}] — {len(self.partitions[name].qa_pairs)} 条")

    def _get_part_cfg(self, name: str) -> dict:
        return self.config.get("partitions", {}).get(name, {})

    def search(
        self,
        query: str,
        partition: Optional[str] = None,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> dict:
        """
        返回:
        {
            "partition_used": "不具合",
            "candidates_count": 40,
            "top_results": [{"rank":1, "final_score":0.923, "vec_score":0.91, "rrf_score":0.031, "qa":{...}}, ...]
        }
        """
        cfg = self._get_part_cfg(partition or "")
        _top_k = top_k or cfg.get("top_k", 5)
        _candidate_k = candidate_k or cfg.get("candidate_k", 20)

        target = self.partitions.get(partition) if partition else None

        query_vec = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True,
        )
        query_vec = np.array(query_vec, dtype=np.float32)
        query_tokens = tokenize_jp(query)

        if target:
            results = self._search_partition(target, query_vec, query_tokens, _candidate_k)
            source_name = target.name
            raw_candidate_count = min(_candidate_k * 2, len(target.qa_pairs))
        else:
            merged: list[dict] = []
            for name, part in self.partitions.items():
                partial = self._search_partition(part, query_vec, query_tokens, _candidate_k // 2)
                merged.extend(partial)
            merged.sort(key=lambda x: x["final_score"], reverse=True)
            results = merged
            source_name = "全分区"
            raw_candidate_count = sum(len(p.qa_pairs) for p in self.partitions.values())

        top = results[:_top_k]
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
        cfg = self._get_part_cfg(part.name)
        vec_w = cfg.get("vec_weight", 0.70)
        rrf_w = cfg.get("rrf_weight", 0.30)

        vec_results = part.vector_search(query_vec, top_k=candidate_k)
        bm25_results = part.bm25_search(query_tokens, top_k=candidate_k)

        vec_score_map = {idx: score for idx, score in vec_results}
        bm25_raw = {idx: score for idx, score in bm25_results}
        bm25_max = max(bm25_raw.values(), default=1.0)
        bm25_norm = {idx: s / bm25_max for idx, s in bm25_raw.items()}

        rrf = rrf_fusion(
            [[i for i, _ in vec_results], [i for i, _ in bm25_results]]
        )
        rrf_max = rrf[0][1] if rrf else 1.0
        rrf_norm = {idx: s / rrf_max for idx, s in rrf[:candidate_k * 2]}

        candidates = set(vec_score_map.keys()) | set(bm25_raw.keys())

        items = []
        for idx in candidates:
            if idx >= len(part.qa_pairs):
                continue
            qa = part.qa_pairs[idx]
            vs = vec_score_map.get(idx, 0.0)
            rs = rrf_norm.get(idx, 0.0)
            score = vec_w * vs + rrf_w * rs
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

    def route(self, confidence: float, turns: int, partition: str = "") -> str:
        """per-partition 路由阈值（从 retrieval_config.json 读取）"""
        cfg = self._get_part_cfg(partition)
        auto_t = cfg.get("auto_threshold", 0.90)
        review_t = cfg.get("review_threshold", 0.70)
        if turns > 1:
            return "HUMAN"
        if confidence >= auto_t:
            return "AUTO"
        if confidence >= review_t:
            return "REVIEW"
        return "HUMAN"
