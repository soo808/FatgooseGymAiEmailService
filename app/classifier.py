"""
classifier.py — 用 DeepSeek 对输入邮件做意图分类，返回分区名
分区: 不具合 | 意見要望 | 購入 | その他
"""

import os

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
