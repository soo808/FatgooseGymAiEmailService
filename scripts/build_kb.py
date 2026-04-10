#!/usr/bin/env python
"""
build_kb.py — 从邮件 Excel 构建知识库（一次性运行）

步骤:
  1. 读取 Excel，按 メールID + 枝番 分组
  2. 提取 QA 对（枝番1 问题 + 枝番2 回复）
  3. 文本清洗
  4. 百度翻译 API（jp → zh）翻译问答对
  5. multilingual-e5-base 生成嵌入向量
  6. 构建分区 FAISS + BM25 索引
  7. 保存 kb/partitions/<分区>/qa_pairs.json + faiss.index + bm25.pkl

用法:
  python scripts/build_kb.py --input D:/agenticRAG/RAW-emailQA-DATA.xlsx
  python scripts/build_kb.py --input D:/agenticRAG/RAW-emailQA-DATA-EXAMPLE.csv.xlsx --test

参数:
  --input     源 Excel 文件路径
  --rate      翻译 API 请求间隔秒数（默认 1.1，标准免费账户安全值；认证账户可设 0.15）
  --test      仅处理前 50 条 QA 对（用于快速验证）
  --skip-translate  跳过翻译（直接用日文，用于调试向量索引）
"""

import argparse
import hashlib
import http.client
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.parse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")


def _make_ssl_context() -> ssl.SSLContext:
    """
    创建绕过代理和 SSL EOF 问题的上下文。
    - 不走系统代理（http.client 默认不读取代理环境变量）
    - 关闭证书验证（百度 API 服务端 TLS 握手不规范，Python 3.12+ 会抛 EOF）
    - 设置 OP_IGNORE_UNEXPECTED_EOF（Python 3.12+）
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    if hasattr(ssl, 'OP_IGNORE_UNEXPECTED_EOF'):
        ctx.options |= ssl.OP_IGNORE_UNEXPECTED_EOF
    return ctx


_SSL_CTX = _make_ssl_context()


def _safe_str(val) -> str:
    """将 Excel 单元格值转为字符串，NaN/None 返回空字符串"""
    if val is None:
        return ""
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return ""
    except Exception:
        pass
    return str(val)


BAIDU_APPID = os.getenv("BAIDU_APPID", "")
BAIDU_SECRET = os.getenv("BAIDU_SECRET", "")

KB_DIR = Path(__file__).parent.parent / "kb"
CACHE_FILE = KB_DIR / "translation_cache.json"

# ── 分区映射（inquiry_type → partition name）──
PARTITION_MAP = {
    "不具合":  "不具合",
    "ご意見":  "意見要望",
    "意見":    "意見要望",
    "購入":    "購入",
}


def get_partition(inquiry_type: str) -> str:
    for key, val in PARTITION_MAP.items():
        if key in inquiry_type:
            return val
    return "その他"


# ─────────────────────────────────────────────────────────────────────────────
# 1. 数据提取 & 清洗
# ─────────────────────────────────────────────────────────────────────────────

def extract_question_fields(text: str) -> tuple[str, str, str, str, str]:
    """从枝番1正文中提取核心问题、问题类型、使用环境、发生日期、App版本"""
    if not text:
        return "", "", "", "", ""

    # 核心问题（优先提取结构化字段）
    m = re.search(r'お問い合わせ内容\s*[：:]\s*(.+?)(?:\n添付|$)', text, re.DOTALL)
    question = m.group(1).strip() if m else text.strip()

    # 去掉多余空行
    question = re.sub(r'\n{3,}', '\n\n', question).strip()

    # 问题类型
    m = re.search(r'お問い合わせ内容の種類\s*[：:]\s*(.+)', text)
    inquiry_type = m.group(1).strip() if m else ""

    # 使用环境
    m = re.search(r'ご利用環境\s*[：:]\s*(.+)', text)
    environment = m.group(1).strip() if m else ""

    # 问题发生日期（格式: 2024-03-31）
    m = re.search(r'問題が発生した日時\s*[：:]\s*(\d{4}-\d{2}-\d{2})', text)
    date_str = m.group(1) if m else ""

    # App 版本
    m = re.search(r'アプリバージョン\s*[：:]\s*([\d.]+)', text)
    app_version = m.group(1).strip() if m else ""

    return question, inquiry_type, environment, date_str, app_version


def clean_reply(text: str) -> str:
    """清洗枝番2回复文本：去掉引用行、转发头、免责声明"""
    if not text:
        return ""

    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # 遇到引用行（>开头）停止
        if line.startswith('>'):
            break
        # 遇到转发时间戳停止（如 2024年3月21日(木)）
        if re.match(r'\d{4}年\d{1,2}月\d{1,2}日', line.strip()):
            break
        # 跳过免责声明行
        if '無断掲載' in line or '無断複製' in line or '転送はお控え' in line:
            continue
        cleaned.append(line)

    return '\n'.join(cleaned).strip()


def count_turns(df_thread: pd.DataFrame) -> int:
    """统计邮件线程的轮次数"""
    return int(df_thread['メールID枝番'].max())


def extract_qa_pairs(df: pd.DataFrame) -> list[dict]:
    """从 DataFrame 提取所有 QA 对"""
    qa_pairs = []

    for mail_id, group in df.groupby('メールID'):
        group = group.sort_values('メールID枝番')
        turns = {int(row['メールID枝番']): row['本文'] for _, row in group.iterrows()}
        total_turns = max(turns.keys())

        # 必须有枝番1和枝番2
        if 1 not in turns or 2 not in turns:
            continue

        question, inquiry_type, environment, date_str, app_version = \
            extract_question_fields(_safe_str(turns[1]))
        answer = clean_reply(_safe_str(turns[2]))

        # 过滤空内容
        if not question or not answer:
            continue

        qa_pairs.append({
            "id": str(mail_id),
            "question_jp": question,
            "answer_jp": answer,
            "question_zh": "",   # 翻译后填入
            "answer_zh": "",     # 翻译后填入
            "inquiry_type": inquiry_type,
            "environment": environment,
            "app_version": app_version,
            "date": date_str,
            "total_turns": total_turns,
            "partition": get_partition(inquiry_type),
        })

    return qa_pairs


# ─────────────────────────────────────────────────────────────────────────────
# 2. 百度翻译 API
# ─────────────────────────────────────────────────────────────────────────────

def baidu_translate(text: str, from_lang: str = "jp", to_lang: str = "zh") -> str:
    """
    调用百度翻译通用文本 API（标准版）。
    使用 http.client 直连，绕过系统代理和 urllib3/requests 的 SSL 处理层。
    签名规则: MD5(appid + q + salt + secret)，q 不做 URL 编码。
    """
    if not BAIDU_APPID or not BAIDU_SECRET or BAIDU_APPID.startswith("请"):
        raise EnvironmentError(
            "百度翻译 API 未配置，请在 .env 中填入 BAIDU_APPID 和 BAIDU_SECRET"
        )

    salt = str(random.randint(100000, 999999))
    sign_str = BAIDU_APPID + text + salt + BAIDU_SECRET
    sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()

    payload = {
        'q': text,
        'from': from_lang,
        'to': to_lang,
        'appid': BAIDU_APPID,
        'salt': salt,
        'sign': sign,
    }

    # URL 编码表单体（q 在签名时已使用原始值，这里编码用于 HTTP 传输）
    body = urllib.parse.urlencode(payload).encode('utf-8')

    # http.client 直连，不读取 HTTP_PROXY/HTTPS_PROXY 环境变量
    conn = http.client.HTTPSConnection(
        'fanyi-api.baidu.com',
        timeout=15,
        context=_SSL_CTX,
    )
    try:
        conn.request(
            'POST',
            '/api/trans/vip/translate',
            body=body,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
        )
        response = conn.getresponse()
        raw = response.read().decode('utf-8')
    finally:
        conn.close()

    result = json.loads(raw)

    if 'trans_result' in result:
        return '\n'.join(item['dst'] for item in result['trans_result'])
    else:
        error_code = result.get('error_code', 'unknown')
        error_msg = result.get('error_msg', '')
        raise ValueError(f"翻译失败 [{error_code}] {error_msg}")


def translate_with_retry(text: str, max_retries: int = 3, rate: float = 1.1) -> str:
    """带重试和速率限制的翻译函数"""
    for attempt in range(max_retries):
        try:
            result = baidu_translate(text)
            time.sleep(rate)
            return result
        except ValueError as e:
            err = str(e)
            if '20003' in err:
                # 命中敏感词，不可重试，跳过并返回占位符
                time.sleep(rate)
                return "[内容含敏感词，翻译已跳过]"
            elif '54004' in err:
                # 月度字符超限，继续重试已无意义
                raise RuntimeError(
                    "百度翻译月度免费额度已耗尽（54004），"
                    "请在控制台开启按量付费后重新运行（支持断点续传）。"
                ) from e
            elif '54003' in err:
                # 频率超限，等待后重试
                time.sleep(rate * 3)
            elif attempt == max_retries - 1:
                raise
            else:
                time.sleep(rate * 2)
    return ""


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    KB_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def translate_all(qa_pairs: list[dict], rate: float = 1.1) -> list[dict]:
    """翻译所有 QA 对，支持断点续传"""
    cache = load_cache()
    print(f"\n[翻译] 共 {len(qa_pairs)} 对，已缓存 {len(cache)} 条，速率 {rate}s/次")

    for i, qa in enumerate(tqdm(qa_pairs, desc="翻译进度")):
        qa_id = qa['id']
        cache_key_q = f"{qa_id}_q"
        cache_key_a = f"{qa_id}_a"

        # 翻译问题（断点续传：缓存中已有则跳过）
        if cache_key_q in cache:
            qa['question_zh'] = cache[cache_key_q]
        else:
            qa['question_zh'] = translate_with_retry(qa['question_jp'], rate=rate)
            cache[cache_key_q] = qa['question_zh']

        # 翻译回复
        if cache_key_a in cache:
            qa['answer_zh'] = cache[cache_key_a]
        else:
            qa['answer_zh'] = translate_with_retry(qa['answer_jp'], rate=rate)
            cache[cache_key_a] = qa['answer_zh']

        # 每 50 条保存一次缓存
        if (i + 1) % 50 == 0:
            save_cache(cache)

    save_cache(cache)
    return qa_pairs


# ─────────────────────────────────────────────────────────────────────────────
# 3. 向量嵌入 + FAISS + BM25 分区索引
# ─────────────────────────────────────────────────────────────────────────────

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
        print("请运行: pip install sentence-transformers faiss-cpu rank_bm25")
        sys.exit(1)

    print("\n[向量] 加载 intfloat/multilingual-e5-base（首次运行会自动下载 ~280MB）")
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="构建知识库")
    parser.add_argument('--input', required=True, help='源 Excel 文件路径')
    parser.add_argument('--rate', type=float, default=1.1,
                        help='翻译 API 请求间隔(秒)。默认1.1(免费账户)；认证账户可设0.15')
    parser.add_argument('--test', action='store_true',
                        help='仅处理前50条 QA 对（快速验证）')
    parser.add_argument('--skip-translate', action='store_true',
                        help='跳过翻译（调试向量索引用）')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[错误] 文件不存在: {input_path}")
        sys.exit(1)

    # ── 读取 Excel ──
    print(f"[读取] {input_path.name} ...")
    df = pd.read_excel(input_path, dtype={'メールID': str, 'メールID枝番': int})
    print(f"       {len(df)} 行 / {df['メールID'].nunique()} 个邮件线程")

    # ── 提取 QA 对 ──
    print("[提取] 清洗并提取 QA 对...")
    qa_pairs = extract_qa_pairs(df)
    print(f"       有效 QA 对: {len(qa_pairs)} 条")

    if args.test:
        qa_pairs = qa_pairs[:50]
        print(f"[测试] 已截断至前 50 条")

    if not qa_pairs:
        print("[错误] 没有提取到任何 QA 对，请检查 Excel 格式")
        sys.exit(1)

    # ── 翻译 ──
    if args.skip_translate:
        print("[跳过] 翻译阶段（--skip-translate）")
        for qa in qa_pairs:
            qa['question_zh'] = qa['question_jp']
            qa['answer_zh'] = qa['answer_jp']
    else:
        qa_pairs = translate_all(qa_pairs, rate=args.rate)

    # ── 按分区构建索引 ──
    partitions: dict[str, list[dict]] = defaultdict(list)
    for qa in qa_pairs:
        partitions[qa["partition"]].append(qa)

    print(f"\n[分区] 共 {len(partitions)} 个分区：")
    for name, items in partitions.items():
        print(f"  {name}: {len(items)} 条")

    build_partitioned_index(dict(partitions))

    print("\n✓ 知识库构建完成！")
    print("  kb/partitions/<分区>/qa_pairs.json  — 各分区问答对（含中文译文）")
    print("  kb/partitions/<分区>/faiss.index    — FAISS 向量索引")
    print("  kb/partitions/<分区>/bm25.pkl       — BM25 索引")
    print("\n下一步:")
    print("  python scripts/analyze_kb.py")
    print("  uvicorn app.main:app --reload --app-dir D:/agenticRAG")


if __name__ == '__main__':
    main()
