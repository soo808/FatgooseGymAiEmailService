"""
routers/kb.py — 知识库管理 API
挂载到 /api/kb（在 main.py 中 include_router）

端点：
  GET  /partitions                  — 分区列表 + 条数 + 配置摘要
  GET  /entries/{partition}         — 分页列表
  GET  /entries/{partition}/{id}    — 单条完整 QA
  PUT  /entries/{partition}/{id}    — 更新字段（partial update）
  DELETE /entries/{partition}/{id} — 删除
  POST /entries/{partition}         — 新增手动条目
  POST /rebuild/{partition}         — SSE 流：重建索引进度
  POST /import                      — multipart 文件导入，SSE 进度
  GET  /config                      — 读 retrieval_config.json
  GET  /config/{partition}          — 单分区配置
  PUT  /config/{partition}          — 更新配置（验证 vec+rrf=1.0）
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import kb_manager, file_importer

router = APIRouter()

KB_DIR = Path(__file__).parent.parent.parent / "kb"
CONFIG_PATH = KB_DIR / "retrieval_config.json"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"partitions": {}}


def _save_config(cfg: dict) -> None:
    import os
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# 分区列表
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/partitions")
async def list_partitions():
    cfg = _load_config()
    parts_dir = KB_DIR / "partitions"
    result = []

    # 已知分区（含配置）
    known = set(cfg.get("partitions", {}).keys())

    # 磁盘上存在的分区
    disk_parts: set[str] = set()
    if parts_dir.exists():
        for d in parts_dir.iterdir():
            if d.is_dir() and (d / "qa_pairs.json").exists():
                disk_parts.add(d.name)

    all_parts = known | disk_parts
    for name in sorted(all_parts):
        entries = kb_manager._load_qa(name)
        part_cfg = cfg.get("partitions", {}).get(name, {})
        result.append({
            "name": name,
            "count": len(entries),
            "description": part_cfg.get("description", ""),
            "vec_weight": part_cfg.get("vec_weight", 0.70),
            "rrf_weight": part_cfg.get("rrf_weight", 0.30),
            "top_k": part_cfg.get("top_k", 5),
            "auto_threshold": part_cfg.get("auto_threshold", 0.90),
            "review_threshold": part_cfg.get("review_threshold", 0.70),
        })

    total = sum(r["count"] for r in result)
    return {"partitions": result, "total": total}


# ─────────────────────────────────────────────────────────────────────────────
# 条目 CRUD
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/entries/{partition}")
async def list_entries(partition: str, page: int = 1, page_size: int = 20):
    return kb_manager.list_entries(partition, page=page, page_size=page_size)


@router.get("/entries/{partition}/{entry_id}")
async def get_entry(partition: str, entry_id: str):
    entry = kb_manager.get_entry(partition, entry_id)
    if entry is None:
        raise HTTPException(404, f"条目 {entry_id} 不存在于分区 {partition}")
    return entry


class UpdateEntryBody(BaseModel):
    question_jp: Optional[str] = None
    answer_jp: Optional[str] = None
    question_zh: Optional[str] = None
    answer_zh: Optional[str] = None
    inquiry_type: Optional[str] = None
    environment: Optional[str] = None
    app_version: Optional[str] = None
    date: Optional[str] = None


@router.put("/entries/{partition}/{entry_id}")
async def update_entry(partition: str, entry_id: str, body: UpdateEntryBody):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    entry = kb_manager.update_entry(partition, entry_id, updates)
    if entry is None:
        raise HTTPException(404, f"条目 {entry_id} 不存在于分区 {partition}")
    return entry


@router.delete("/entries/{partition}/{entry_id}")
async def delete_entry(partition: str, entry_id: str):
    ok = kb_manager.delete_entry(partition, entry_id)
    if not ok:
        raise HTTPException(404, f"条目 {entry_id} 不存在于分区 {partition}")
    return {"deleted": entry_id}


class AddEntryBody(BaseModel):
    question_jp: str
    answer_jp: str
    question_zh: str = ""
    answer_zh: str = ""
    inquiry_type: str = ""
    environment: str = ""
    app_version: str = ""
    date: str = ""


@router.post("/entries/{partition}")
async def add_entry(partition: str, body: AddEntryBody):
    entry = kb_manager.add_entry(partition, body.model_dump())
    return entry


# ─────────────────────────────────────────────────────────────────────────────
# 重建索引（SSE 进度流）
# ─────────────────────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/rebuild/{partition}")
async def rebuild_partition(partition: str):
    """SSE 流：重建指定分区的索引"""

    async def stream():
        entries = kb_manager._load_qa(partition)
        count = len(entries)
        yield _sse("start", {"count": count})

        progress_q: asyncio.Queue = asyncio.Queue()

        def progress_cb(done: int, total: int):
            pct = int(done * 100 / total) if total else 0
            asyncio.get_event_loop().call_soon_threadsafe(
                progress_q.put_nowait, pct
            )

        t0 = time.time()
        rebuild_task = asyncio.create_task(
            kb_manager.rebuild_index_for_partition(partition, progress_cb)
        )

        while not rebuild_task.done():
            try:
                pct = await asyncio.wait_for(progress_q.get(), timeout=0.5)
                yield _sse("progress", {"pct": pct})
            except asyncio.TimeoutError:
                pass

        try:
            await rebuild_task
        except Exception as e:
            yield _sse("error", {"message": str(e)})
            return

        elapsed = int((time.time() - t0) * 1000)
        yield _sse("done", {"elapsed_ms": elapsed})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 文件导入（SSE 进度流）
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/import")
async def import_file(
    file: UploadFile = File(...),
    partition: str = Form(...),
    description: str = Form(""),
    parse_mode: str = Form("qa_excel"),
    q_col: str = Form("質問"),
    a_col: str = Form("回答"),
    text_col: str = Form("本文"),
    q_marker: str = Form("Q:"),
    a_marker: str = Form("A:"),
):
    """
    multipart 文件导入，SSE 进度流。
    parse_mode:
      qa_excel        — Excel Q/A 双列
      paragraph_excel — Excel 单列段落
      docx_qa         — docx Q/A 标记行
      docx_paragraphs — docx Heading/段落
    """
    file_bytes = await file.read()

    async def stream():
        yield _sse("start", {"filename": file.filename, "partition": partition})

        try:
            if parse_mode == "qa_excel":
                entries = await asyncio.to_thread(
                    file_importer.parse_qa_excel, file_bytes, q_col, a_col
                )
            elif parse_mode == "paragraph_excel":
                entries = await asyncio.to_thread(
                    file_importer.parse_paragraph_excel, file_bytes, text_col
                )
            elif parse_mode == "docx_qa":
                entries = await asyncio.to_thread(
                    file_importer.parse_docx_qa, file_bytes, q_marker, a_marker
                )
            elif parse_mode == "docx_paragraphs":
                entries = await asyncio.to_thread(
                    file_importer.parse_docx_paragraphs, file_bytes
                )
            else:
                yield _sse("error", {"message": f"未知解析模式: {parse_mode}"})
                return

            yield _sse("parsed", {"count": len(entries)})

            # 写入分区 + 更新 config
            for entry in entries:
                entry["partition"] = partition
                kb_manager.add_entry(partition, entry)

            # 将新分区加入 retrieval_config
            await asyncio.to_thread(
                kb_manager.add_partition_to_config, partition, description
            )

            yield _sse("progress", {"pct": 50, "message": "条目已写入，开始重建索引..."})

            # 重建索引
            progress_q: asyncio.Queue = asyncio.Queue()

            def progress_cb(done: int, total: int):
                pct = 50 + int(done * 50 / total) if total else 50
                asyncio.get_event_loop().call_soon_threadsafe(
                    progress_q.put_nowait, pct
                )

            rebuild_task = asyncio.create_task(
                kb_manager.rebuild_index_for_partition(partition, progress_cb)
            )

            while not rebuild_task.done():
                try:
                    pct = await asyncio.wait_for(progress_q.get(), timeout=0.5)
                    yield _sse("progress", {"pct": pct})
                except asyncio.TimeoutError:
                    pass

            await rebuild_task

            # 热更新分类器配置
            try:
                from app.retriever import HybridRetriever
                from app import kb_manager as _km
                if _km._retriever is not None:
                    _km._retriever.reload_config()
            except Exception:
                pass

            yield _sse("done", {"partition": partition, "imported": len(entries)})

        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 检索配置
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/config")
async def get_config():
    return _load_config()


@router.get("/config/{partition}")
async def get_partition_config(partition: str):
    cfg = _load_config()
    part_cfg = cfg.get("partitions", {}).get(partition)
    if part_cfg is None:
        raise HTTPException(404, f"分区 {partition} 的配置不存在")
    return part_cfg


class PartitionConfigBody(BaseModel):
    description: Optional[str] = None
    vec_weight: Optional[float] = None
    rrf_weight: Optional[float] = None
    candidate_k: Optional[int] = None
    top_k: Optional[int] = None
    auto_threshold: Optional[float] = None
    review_threshold: Optional[float] = None


@router.put("/config/{partition}")
async def update_partition_config(partition: str, body: PartitionConfigBody):
    cfg = _load_config()
    parts = cfg.setdefault("partitions", {})
    existing = parts.get(partition, {})

    updates = {k: v for k, v in body.model_dump().items() if v is not None}

    merged = {**existing, **updates}

    # 验证权重之和 = 1.0（允许 ±0.01 浮点误差）
    vec = merged.get("vec_weight", 0.70)
    rrf = merged.get("rrf_weight", 0.30)
    if abs(vec + rrf - 1.0) > 0.01:
        raise HTTPException(
            422,
            f"vec_weight({vec}) + rrf_weight({rrf}) = {vec + rrf:.3f}，必须等于 1.0"
        )

    parts[partition] = merged
    _save_config(cfg)

    # 热更新 retriever 配置
    from app import kb_manager as _km
    if _km._retriever is not None:
        _km._retriever.reload_config()

    return merged
