"""
main.py — FastAPI 后端 v3
  GET  /                → 前端页面
  POST /api/process     → SSE 流式处理邮件
  GET  /api/analytics   → 返回 analytics.json
  GET  /api/metrics     → 返回检索质量指标（聚合 retrieval_log.jsonl）
  GET  /api/health      → 健康检查
  /api/kb/*             → 知识库管理（include kb_router）
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
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
from app import kb_manager
from app.routers.kb import router as kb_router

app = FastAPI(title="ぽちゃガチョ！客服 AI v3")
app.include_router(kb_router, prefix="/api/kb", tags=["kb"])

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_retriever: HybridRetriever | None = None
KB_DIR = Path(__file__).parent.parent / "kb"
RETRIEVAL_LOG = KB_DIR / "retrieval_log.jsonl"


@app.on_event("startup")
async def startup():
    global _retriever
    try:
        _retriever = HybridRetriever()
        kb_manager.set_retriever(_retriever)
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
    return {"status": "ok", "kb_ready": kb_ready, "kb_size": total, "total": total, "partitions": partitions}


@app.get("/api/analytics")
async def analytics():
    path = KB_DIR / "analytics.json"
    if not path.exists():
        raise HTTPException(404, "analytics.json 不存在，请运行 scripts/analyze_kb.py")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@app.get("/api/metrics")
async def metrics():
    """
    聚合 retrieval_log.jsonl，返回检索质量指标。
    - 置信度分布（4个区间）按分区分组
    - 路由等级分布（AUTO/REVIEW/HUMAN）
    - 每日请求量（最近 30 天）
    """
    if not RETRIEVAL_LOG.exists():
        return {
            "total_requests": 0,
            "confidence_dist": {},
            "level_dist": {"AUTO": 0, "REVIEW": 0, "HUMAN": 0},
            "daily_requests": [],
            "avg_top1_score": 0,
        }

    records = []
    with open(RETRIEVAL_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return {
            "total_requests": 0,
            "confidence_dist": {},
            "level_dist": {"AUTO": 0, "REVIEW": 0, "HUMAN": 0},
            "daily_requests": [],
            "avg_top1_score": 0,
        }

    # 置信度分布（4 个区间）按分区分组
    conf_dist: dict[str, dict[str, int]] = defaultdict(lambda: {
        "≥90%": 0, "70–90%": 0, "50–70%": 0, "<50%": 0
    })
    level_dist: dict[str, int] = defaultdict(int)
    daily: dict[str, int] = defaultdict(int)
    top1_scores = []

    for r in records:
        conf = r.get("confidence", 0)
        part = r.get("partition", "その他")
        level = r.get("level", "HUMAN")
        ts = r.get("ts", "")[:10]  # yyyy-mm-dd
        top1 = r.get("top1_score", 0)

        if conf >= 90:
            conf_dist[part]["≥90%"] += 1
        elif conf >= 70:
            conf_dist[part]["70–90%"] += 1
        elif conf >= 50:
            conf_dist[part]["50–70%"] += 1
        else:
            conf_dist[part]["<50%"] += 1

        level_dist[level] += 1
        if ts:
            daily[ts] += 1
        if top1:
            top1_scores.append(top1)

    # 最近 30 天日志
    from datetime import date, timedelta
    today = date.today()
    daily_list = []
    for i in range(29, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        daily_list.append({"date": day, "count": daily.get(day, 0)})

    return {
        "total_requests": len(records),
        "confidence_dist": dict(conf_dist),
        "level_dist": dict(level_dist),
        "daily_requests": daily_list,
        "avg_top1_score": round(sum(top1_scores) / len(top1_scores), 4) if top1_scores else 0,
    }


def _log_retrieval(
    partition: str,
    confidence: float,
    level: str,
    top_results: list[dict],
    latency_ms: int,
) -> None:
    """写一行到 retrieval_log.jsonl（BackgroundTask）"""
    from datetime import datetime, timezone
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "partition": partition,
        "confidence": round(confidence * 100, 1),
        "level": level,
        "top1_score": top_results[0]["final_score"] if top_results else 0,
        "top5_scores": [r["final_score"] for r in top_results[:5]],
        "latency_ms": latency_ms,
    }
    KB_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRIEVAL_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ProcessRequest(BaseModel):
    email_text: str


@app.post("/api/process")
async def process_email(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    SSE 流式接口。事件格式:
      event: translate  data: {"email_zh": "中文译文..."}
      event: intent     data: {"partition": "不具合"}
      event: meta       data: {"confidence": 87.2, "level": "REVIEW", ...}
      event: section    data: {"name": "summary"|"reply"}
      event: token      data: {"text": "...", "section": "summary"|"reply"}
      event: reply_zh   data: {"text": "完整中文译文"}
      event: done       data: {}
      event: error      data: {"message": "..."}
    """
    email_text = req.email_text.strip()
    if not email_text:
        raise HTTPException(400, "邮件内容不能为空")
    if _retriever is None:
        raise HTTPException(503, "知识库未加载，请先运行 build_kb.py")

    async def stream():
        try:
            t_start = time.time()

            # 1. 并行：邮件翻译 + 意图分类
            email_zh, partition = await asyncio.gather(
                translate_email(email_text),
                classify_intent(email_text),
            )
            yield f"event: translate\ndata: {json.dumps({'email_zh': email_zh}, ensure_ascii=False)}\n\n"
            yield f"event: intent\ndata: {json.dumps({'partition': partition}, ensure_ascii=False)}\n\n"

            # 2. 混合检索 + 重排
            turns = HybridRetriever.detect_turns(email_text)
            search_result = _retriever.search(email_text, partition=partition)
            top_results = search_result["top_results"]

            confidence = top_results[0]["vec_score"] if top_results else 0.0
            level = _retriever.route(confidence, turns, partition=partition)

            latency_ms = int((time.time() - t_start) * 1000)

            # 序列化 top_results（展示字段）
            top_for_frontend = [
                {
                    "rank": r["rank"],
                    "score": r["vec_score"],
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

            # 3. 异步记录日志（非阻塞）
            background_tasks.add_task(
                _log_retrieval,
                search_result["partition_used"],
                confidence,
                level,
                top_results,
                latency_ms,
            )

            # 4. LLM 生成
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

            # 5. 翻译日文回复为中文
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
