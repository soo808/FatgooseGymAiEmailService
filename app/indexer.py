"""
indexer.py — FAISS + BM25 分区索引构建的单一入口
  - model singleton（只加载一次 ~280MB）
  - rebuild_partition_index() 原子写入（.tmp → rename）
  - tokenize_jp() 统一 bigram 分词（retriever.py 从这里导入）
"""

import os
import pickle
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_model = None


def get_model():
    """SentenceTransformer singleton，进程内只加载一次"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('intfloat/multilingual-e5-base')
    return _model


def tokenize_jp(text: str) -> list[str]:
    """日文字符 bigram 分词，与 build_kb.py 保持一致"""
    text = re.sub(r'\s+', '', text)
    chars = list(text)
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return chars + bigrams


def rebuild_partition_index(
    partition_name: str,
    qa_list: list[dict],
    kb_dir: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    为单个分区重建 FAISS + BM25 索引（原子写入）。
    写临时文件后 os.replace() 保证不出现半写状态。

    Args:
        partition_name: 分区名称（如 "不具合"）
        qa_list: 该分区全部 QA 对列表
        kb_dir: kb/ 根目录 Path
        progress_cb: 可选进度回调 progress_cb(done, total)
    """
    import faiss
    from rank_bm25 import BM25Okapi

    out_dir = kb_dir / "partitions" / partition_name
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(qa_list)
    model = get_model()

    # ── 1. 保存 QA JSON（原子写入）──
    qa_tmp = out_dir / "qa_pairs.json.tmp"
    import json
    with open(qa_tmp, 'w', encoding='utf-8') as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)
    os.replace(qa_tmp, out_dir / "qa_pairs.json")

    if progress_cb:
        progress_cb(0, total)

    # ── 2. FAISS 向量索引 ──
    questions = [qa.get("question_jp", "") for qa in qa_list]
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

    faiss_tmp = out_dir / "faiss.index.tmp"
    faiss.write_index(index, str(faiss_tmp))
    os.replace(faiss_tmp, out_dir / "faiss.index")

    if progress_cb:
        progress_cb(total // 2, total)

    # ── 3. BM25 索引 ──
    tokenized = [tokenize_jp(q) for q in questions]
    bm25 = BM25Okapi(tokenized)

    bm25_tmp = out_dir / "bm25.pkl.tmp"
    with open(bm25_tmp, 'wb') as f:
        pickle.dump(bm25, f)
    os.replace(bm25_tmp, out_dir / "bm25.pkl")

    if progress_cb:
        progress_cb(total, total)
