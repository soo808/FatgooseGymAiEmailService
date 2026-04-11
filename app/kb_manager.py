"""
kb_manager.py — 知识库 CRUD 业务逻辑
  - list_entries / get_entry / update_entry / delete_entry / add_entry
  - rebuild_index_for_partition（asyncio.to_thread，不阻塞事件循环）
  - 每个分区一把 asyncio.Lock，防止并发重建文件损坏
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from app import indexer as _indexer

KB_DIR = Path(__file__).parent.parent / "kb"

_retriever = None
_rebuild_locks: dict[str, asyncio.Lock] = {}


def set_retriever(r) -> None:
    """main.py startup 时调用，注入 HybridRetriever 实例"""
    global _retriever
    _retriever = r


def _lock_for(partition: str) -> asyncio.Lock:
    if partition not in _rebuild_locks:
        _rebuild_locks[partition] = asyncio.Lock()
    return _rebuild_locks[partition]


def _qa_path(partition: str) -> Path:
    return KB_DIR / "partitions" / partition / "qa_pairs.json"


def _load_qa(partition: str) -> list[dict]:
    p = _qa_path(partition)
    if not p.exists():
        return []
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_entries(
    partition: str,
    page: int = 1,
    page_size: int = 20,
    filter_partition: Optional[str] = None,
) -> dict:
    """
    返回分页数据。
    如果 partition == "全部"，合并所有分区（保留各条目的 partition 字段）。
    按 date DESC 排序（空 date 排末尾）。
    """
    if partition == "全部":
        # 合并所有分区
        parts_dir = KB_DIR / "partitions"
        all_entries: list[dict] = []
        if parts_dir.exists():
            for d in parts_dir.iterdir():
                if d.is_dir() and (d / "qa_pairs.json").exists():
                    for qa in _load_qa(d.name):
                        qa.setdefault("partition", d.name)
                        all_entries.append(qa)
        entries = all_entries
    else:
        entries = _load_qa(partition)
        for qa in entries:
            qa.setdefault("partition", partition)

    # 按 date DESC（空排末尾）
    def sort_key(qa):
        d = qa.get("date", "")
        return ("" if d else "0") + (d or "")

    entries.sort(key=sort_key, reverse=True)

    total = len(entries)
    start = (page - 1) * page_size
    end = start + page_size
    page_data = entries[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "entries": page_data,
    }


def get_entry(partition: str, entry_id: str) -> Optional[dict]:
    for qa in _load_qa(partition):
        if str(qa.get("id", "")) == entry_id:
            qa.setdefault("partition", partition)
            return qa
    return None


def update_entry(partition: str, entry_id: str, updates: dict) -> Optional[dict]:
    """Partial update；保留未传字段。返回更新后的条目，不存在返回 None"""
    entries = _load_qa(partition)
    for i, qa in enumerate(entries):
        if str(qa.get("id", "")) == entry_id:
            # 不允许外部修改 id / partition
            updates.pop("id", None)
            updates.pop("partition", None)
            entries[i] = {**qa, **updates}
            _save_qa(partition, entries)
            return entries[i]
    return None


def delete_entry(partition: str, entry_id: str) -> bool:
    """删除条目。成功返回 True，未找到返回 False"""
    entries = _load_qa(partition)
    before = len(entries)
    entries = [qa for qa in entries if str(qa.get("id", "")) != entry_id]
    if len(entries) == before:
        return False
    _save_qa(partition, entries)
    return True


def add_entry(partition: str, entry: dict) -> dict:
    """
    新增条目。
    ID 策略：现有 ID 全为整数 → max+1；否则 uuid4。
    如果分区目录不存在则创建。
    """
    entries = _load_qa(partition)

    # 分配 ID
    int_ids = []
    for qa in entries:
        try:
            int_ids.append(int(qa.get("id", "")))
        except (ValueError, TypeError):
            pass

    new_id = str(max(int_ids) + 1) if int_ids else str(uuid.uuid4())

    new_entry = {
        "id": new_id,
        "question_jp": entry.get("question_jp", ""),
        "answer_jp": entry.get("answer_jp", ""),
        "question_zh": entry.get("question_zh", ""),
        "answer_zh": entry.get("answer_zh", ""),
        "inquiry_type": entry.get("inquiry_type", partition),
        "environment": entry.get("environment", ""),
        "app_version": entry.get("app_version", ""),
        "date": entry.get("date", ""),
        "total_turns": entry.get("total_turns", 1),
        "partition": partition,
    }
    entries.append(new_entry)

    # 确保分区目录存在
    part_dir = KB_DIR / "partitions" / partition
    part_dir.mkdir(parents=True, exist_ok=True)

    _save_qa(partition, entries)
    return new_entry


def add_partition_to_config(partition_name: str, description: str = "") -> None:
    """
    将新分区写入 retrieval_config.json（导入新文件时调用）。
    如果已存在则不覆盖。
    """
    cfg_path = KB_DIR / "retrieval_config.json"
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    else:
        cfg = {"partitions": {}}

    if partition_name not in cfg["partitions"]:
        cfg["partitions"][partition_name] = {
            "description": description or partition_name,
            "vec_weight": 0.75,
            "rrf_weight": 0.25,
            "candidate_k": 20,
            "top_k": 5,
            "auto_threshold": 0.90,
            "review_threshold": 0.70,
        }
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


def _save_qa(partition: str, entries: list[dict]) -> None:
    import os
    path = _qa_path(partition)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


async def rebuild_index_for_partition(
    partition: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    重建指定分区的 FAISS + BM25 索引（不阻塞事件循环）。
    完成后热更新 HybridRetriever。
    同一分区并发重建时后来的请求等待前者完成。
    """
    lock = _lock_for(partition)
    async with lock:
        entries = _load_qa(partition)
        await asyncio.to_thread(
            _indexer.rebuild_partition_index,
            partition,
            entries,
            KB_DIR,
            progress_cb,
        )
        if _retriever is not None:
            _retriever.reload_partition(partition)
