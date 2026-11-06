"""
日常使用入口:路由 Agent 回答一个问题,打印轨迹。

    python -m router.cli "下周有哪些岗位要面试?"
    python -m router.cli "McNemar 检验怎么算" --config qwen3i-nothink

后端:笔记 = rag/store 的真实索引;岗位 = data/jobs/jobs_snapshot.json(从 Notion 导出,不进仓库);
搜索 = 尚未接入(返回"未配置",模型应如实说明),以后在 router/search.py 加真实后端。
"""

import argparse
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from agent import run_agent  # noqa: E402
from router.jobs import JobsTable, load_jobs  # noqa: E402
from router.tools import SYSTEM_PROMPT, build_tools, to_openai_schemas  # noqa: E402

JOBS_PATH = os.path.join(ROOT, "data", "jobs", "jobs_snapshot.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--config", default="qwen3i-nothink")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from compare_local import COMMON, CONFIGS
    from providers.ollama_provider import make_ollama_decide_fn
    from rag.retrieve import Retriever

    cfg = next(c for c in CONFIGS if c["name"] == args.config)
    jobs = JobsTable(load_jobs(JOBS_PATH)) if os.path.exists(JOBS_PATH) else None
    tools = build_tools(retriever=Retriever.load(cpu=True), jobs_table=jobs, search_backend=None)
    brain = make_ollama_decide_fn(cfg["model"], cfg["think"], {**COMMON, "num_ctx": 8192, **cfg["sampling"]},
                                  user_suffix=cfg.get("user_suffix", ""), tools=to_openai_schemas(),
                                  system=SYSTEM_PROMPT.format(today=date.today().isoformat()))
    traj = run_agent(args.question, decide_fn=brain, tool_functions=tools, max_steps=8, verbose=True)
    traj.print_summary()


if __name__ == "__main__":
    main()
