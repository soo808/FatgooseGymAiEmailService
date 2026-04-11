"""
classifier.py — 用 DeepSeek 对输入邮件做意图分类，返回分区名
分区列表和描述从 kb/retrieval_config.json 动态读取（失败时回退硬编码）。
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_FALLBACK_PARTITIONS = ["不具合", "意見要望", "購入", "その他"]
_FALLBACK_DESCS = {
    "不具合":  "バグ・クラッシュ・データ消失・動作不良",
    "意見要望": "改善提案・新機能要望・感想",
    "購入":    "課金・アイテム未反映・返金",
    "その他":  "上記以外の問い合わせ",
}

_CONFIG_PATH = Path(__file__).parent.parent / "kb" / "retrieval_config.json"

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
    return _client


def _load_partitions_from_config() -> tuple[list[str], dict[str, str]]:
    """
    从 retrieval_config.json 读取分区名称和描述。
    失败时回退到硬编码列表（确保分类器始终可用）。
    """
    try:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            parts = cfg.get("partitions", {})
            if parts:
                names = list(parts.keys())
                descs = {k: v.get("description", k) for k, v in parts.items()}
                # 确保兜底分区 "その他" 始终在末尾
                if "その他" in names:
                    names.remove("その他")
                    names.append("その他")
                return names, descs
    except Exception:
        pass
    return _FALLBACK_PARTITIONS, _FALLBACK_DESCS


async def classify_intent(email_text: str) -> str:
    """
    返回分区名（动态从配置读取）。
    失败时返回 "その他"，不中断主流程。
    """
    names, descs = _load_partitions_from_config()
    bullets = "\n".join(f"- {n}: {descs.get(n, n)}" for n in names)
    valid_choices = "、".join(f"「{n}」" for n in names)

    prompt = f"""以下は日本のゲームサポートに届いたメールです。
内容を読んで、最も当てはまるカテゴリを1つだけ選んでください。

カテゴリ:
{bullets}

メール:
{email_text[:800]}

回答は上記カテゴリ（{valid_choices}）のいずれか1単語のみ。"""

    try:
        resp = await _get_client().chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        raw = resp.choices[0].message.content.strip()
        for p in names:
            if p in raw:
                return p
        return names[-1]  # 兜底：最后一个分区（通常是 "その他"）
    except Exception:
        return names[-1]
