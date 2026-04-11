"""
llm.py — DeepSeek API 调用（演示/推理阶段）
三种路由对应三个流式生成函数，均为 async generator，yield str token
新增 translate_email() / translate_reply() 用于邮件和回复的中文翻译
"""

import os
from typing import AsyncGenerator

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_client = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key or api_key.startswith("请"):
            raise EnvironmentError(
                "DeepSeek API 未配置，请在 .env 中填入 DEEPSEEK_API_KEY"
            )
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    return _client


GAME = "ぽちゃガチョ！"


async def _stream(
    messages: list[dict],
    model: str = "deepseek-chat",
    temperature: float = 0.3,
) -> AsyncGenerator[str, None]:
    client = get_client()
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=1024,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


async def generate_auto_reply(
    question_jp: str,
    top_results: list[dict],
) -> AsyncGenerator[str, None]:
    """
    AUTO 档（置信度≥90%）：直接生成日文回复。
    参考 Top-3 历史案例的语气和结构。
    """
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
                "4. 只输出邮件正文\n\n"
                "⚠️ 重要：回复内容只能基于上述参考案例中的实际信息。\n"
                "禁止添加参考案例中未出现的功能说明、版本说明、补偿政策或具体数值。\n"
                "如有不确定内容请说「確認後にご連絡いたします」，不得自行编造。"
            ),
        },
        {
            "role": "user",
            "content": f"【玩家问题】\n{question_jp}\n\n【参考案例（Top-3）】{refs}",
        },
    ]
    async for token in _stream(messages, temperature=0.1):
        yield token


async def generate_review_draft(
    question_jp: str,
    top_results: list[dict],
) -> AsyncGenerator[str, None]:
    """
    REVIEW 档（70%≤置信度<90%）：生成中文摘要 + 日文回复草稿。
    输出格式（供前端按 [SUMMARY_END] 分段）：
      <中文摘要>
      [SUMMARY_END]
      <日文回复草稿>
    """
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
                "只输出这两部分，无其他说明\n\n"
                "⚠️ 重要：日文草稿内容只能基于上述参考案例中的实际信息。\n"
                "禁止添加参考案例中未出现的功能说明、版本说明、补偿政策或具体数值。\n"
                "如有不确定内容请说「確認後にご連絡いたします」，不得自行编造。"
            ),
        },
        {
            "role": "user",
            "content": f"【玩家原文（日文）】\n{question_jp}\n\n【参考案例（Top-3）】{refs}",
        },
    ]
    async for token in _stream(messages, temperature=0.2):
        yield token


async def generate_human_summary(
    email_text_jp: str,
) -> AsyncGenerator[str, None]:
    """
    HUMAN 档（置信度<70% 或多轮对话）：生成中文对话摘要 + 建议。
    """
    messages = [
        {
            "role": "system",
            "content": (
                f"你是游戏《{GAME}》的AI助手，辅助中国客服处理日文玩家邮件。\n"
                "这封邮件需要人工介入，请用中文提供：\n"
                "1. 【问题摘要】：玩家反映的核心问题（2-3句）\n"
                "2. 【情绪判断】：玩家情绪（平静/不满/焦虑/投诉）\n"
                "3. 【建议处理方向】：给客服的简短建议（1-2条）\n\n"
                "如果邮件包含多轮引用，重点分析最新一条消息。\n"
                "输出格式按上述三点，简洁清晰。"
            ),
        },
        {
            "role": "user",
            "content": f"【邮件内容（日文）】\n{email_text_jp}",
        },
    ]
    async for token in _stream(messages):
        yield token


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
