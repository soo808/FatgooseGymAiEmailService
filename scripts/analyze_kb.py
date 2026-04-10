#!/usr/bin/env python
"""analyze_kb.py — 从分区知识库计算统计数据，输出 kb/analytics.json"""

import json
from collections import defaultdict
from pathlib import Path

KB_DIR = Path(__file__).parent.parent / "kb"
PARTS_DIR = KB_DIR / "partitions"


def load_all_qa() -> list[dict]:
    pairs = []
    for part_dir in PARTS_DIR.iterdir():
        json_path = part_dir / "qa_pairs.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                pairs.extend(json.load(f))
    return pairs


def count(items: list[str]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for x in items:
        c[x or "不明"] += 1
    return dict(sorted(c.items(), key=lambda kv: kv[1], reverse=True))


def main():
    qa_pairs = load_all_qa()
    if not qa_pairs:
        print("[错误] 未找到分区数据，请先运行 build_kb.py")
        return

    # 按分区统计
    by_partition = count([qa["partition"] for qa in qa_pairs])

    # 问询类型分布
    inquiry_dist = count([qa["inquiry_type"] for qa in qa_pairs])

    # 使用环境分布
    env_dist = count([qa["environment"] for qa in qa_pairs])

    # 版本分布（只取主版本号 X.Y）
    def normalize_version(v: str) -> str:
        if not v:
            return "不明"
        parts = v.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else v

    version_dist = count([normalize_version(qa.get("app_version", "")) for qa in qa_pairs])

    # 按月统计（date 字段为 YYYY-MM-DD）
    monthly: dict[str, int] = defaultdict(int)
    for qa in qa_pairs:
        d = qa.get("date", "")
        month = d[:7] if len(d) >= 7 else "不明"
        monthly[month] += 1
    date_monthly = dict(sorted(monthly.items()))

    # 版本 × 分区 交叉分析
    vxp: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for qa in qa_pairs:
        v = normalize_version(qa.get("app_version", ""))
        p = qa.get("partition", "その他")
        vxp[v][p] += 1
    # 只保留 Top-10 版本
    top_versions = list(version_dist.keys())[:10]
    version_x_partition = {v: dict(vxp[v]) for v in top_versions if v in vxp}

    analytics = {
        "total": len(qa_pairs),
        "by_partition": by_partition,
        "inquiry_type_dist": inquiry_dist,
        "environment_dist": env_dist,
        "version_dist": version_dist,
        "date_monthly": date_monthly,
        "version_x_partition": version_x_partition,
    }

    out_path = KB_DIR / "analytics.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(analytics, f, ensure_ascii=False, indent=2)
    print(f"✓ analytics.json 已保存 → {out_path}")
    print(f"  总计: {len(qa_pairs)} 条 | 分区: {list(by_partition.keys())}")


if __name__ == '__main__':
    main()
