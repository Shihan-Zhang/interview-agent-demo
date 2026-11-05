"""
故障注入评测:每道题 × 每种条件(无故障 + 5 类故障) × 每个种子,跑一次路由 Agent。

    # 先用假大脑验证管线(不需要 Ollama,几秒跑完)
    python eval/agent_faults/run_faults.py --brain mock-robust --out eval/results/faults_mock_robust.jsonl
    python eval/agent_faults/run_faults.py --brain mock-naive  --out eval/results/faults_mock_naive.jsonl

    # 真模型(需要 Ollama + 笔记索引 rag/store/)
    python eval/agent_faults/run_faults.py --brain ollama --config qwen3i-nothink --prompt guided --limit 2   # 先试
    python eval/agent_faults/run_faults.py --brain ollama --config qwen3i-nothink --prompt guided

每行记录:题号、条件、种子、轨迹、注入事件、结果分类、用量,以及数据血缘(run_id、索引哈希、提示词版本)。
原始轨迹只存一次,判分可以用 summarize_faults.py --regrade 反复重算。可断点续跑。
"""

import argparse
import json
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from agent import run_agent  # noqa: E402
from faults import CONDITIONS, FaultInjector  # noqa: E402
from grader import classify  # noqa: E402
from router.jobs import JobsTable, load_jobs  # noqa: E402
from router.search import ReplayBackend  # noqa: E402
from router.tools import SYSTEM_PROMPT, SYSTEM_PROMPT_BASE, build_tools, to_openai_schemas  # noqa: E402

TODAY = "2026-11-05"          # 冻结"今天",岗位表里的日期题才可复现
MAX_STEPS = 8                 # 恢复要多几轮:重试 / 换查询,5 轮不够
PROMPTS = {"guided": SYSTEM_PROMPT, "base": SYSTEM_PROMPT_BASE}
FIX = os.path.join(HERE, "fixtures")


class StubRetriever:
    """假大脑跑管线时用:返回一段固定文本。真模型评测必须用真实索引。"""
    manifest = {"chunks_sha256": "stub"}

    def search(self, query, k=5, mode="hybrid"):
        return [{"path": ["测试"], "text": "(占位段落)"}]


def load_tasks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {(r["config"], r["prompt"], r["task_id"], r["condition"], r["seed"])
                for r in map(json.loads, filter(str.strip, f))}


def make_brain(args, task, ollama_cfg):
    if args.brain == "mock-robust":
        from mock_brains import make_robust
        return make_robust(task)
    if args.brain == "mock-naive":
        from mock_brains import make_naive
        return make_naive(task)
    from providers.ollama_provider import make_ollama_decide_fn
    options = {**ollama_cfg["common"], **ollama_cfg["sampling"], "seed": ollama_cfg["seed"]}
    return make_ollama_decide_fn(ollama_cfg["model"], ollama_cfg["think"], options,
                                 user_suffix=ollama_cfg.get("user_suffix", ""),
                                 tools=to_openai_schemas(), system=PROMPTS[args.prompt].format(today=TODAY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="mock-robust", choices=["mock-robust", "mock-naive", "ollama"])
    ap.add_argument("--config", default="qwen3i-nothink", help="eval/compare_local.py 里的配置名")
    ap.add_argument("--prompt", default="guided", choices=list(PROMPTS))
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks_v0.jsonl"))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 n 道题(试跑用)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config_name = args.config if args.brain == "ollama" else args.brain
    out = args.out or os.path.join(ROOT, "eval", "results", f"faults_{config_name}_{args.prompt}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ollama_cfg, retriever = None, None
    if args.brain == "ollama":
        sys.path.insert(0, os.path.join(ROOT, "eval"))
        from compare_local import COMMON, CONFIGS
        cfg = next(c for c in CONFIGS if c["name"] == args.config)
        ollama_cfg = {**cfg, "common": {**COMMON, "num_ctx": 8192}}
        from rag.retrieve import Retriever
        retriever = Retriever.load(cpu=True)
    else:
        retriever = StubRetriever()

    jobs = JobsTable(load_jobs(os.path.join(FIX, "jobs_fixture.json")))
    search = ReplayBackend.load(os.path.join(FIX, "search_replay.jsonl"))
    base_tools = build_tools(retriever=retriever, jobs_table=jobs, search_backend=search)

    tasks = load_tasks(args.tasks)[: args.limit]
    conditions = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    done = done_keys(out)
    run_id = uuid.uuid4().hex[:8]
    lineage = {"run_id": run_id, "today": TODAY, "max_steps": MAX_STEPS,
               "notes_index_sha256": retriever.manifest.get("chunks_sha256"),
               "tasks_file": os.path.basename(args.tasks)}

    todo = [(t, c, s) for s in seeds for t in tasks for c in conditions
            if (config_name, args.prompt, t["id"], c, s) not in done]
    print(f"共 {len(todo)} 个回合待跑(已完成 {len(done)}),输出 {out}")
    with open(out, "a", encoding="utf-8") as f:
        for i, (task, cond, seed) in enumerate(todo, 1):
            search.task_id = task["id"]
            inj = FaultInjector(base_tools, task["tool"], cond)
            if ollama_cfg:
                ollama_cfg["seed"] = seed
            brain = make_brain(args, task, ollama_cfg)
            t0 = time.time()
            traj = run_agent(task["question"], decide_fn=brain, verbose=False,
                             tool_functions=inj.wrap(), max_steps=MAX_STEPS).to_dict()
            outcome = classify(task, traj["steps"][-1], inj.genuine_success())
            tools_used = [s["content"]["tool"] for s in traj["steps"] if s["type"] == "tool_call"]
            row = {"config": config_name, "prompt": args.prompt, "task_id": task["id"], "tool": task["tool"],
                   "condition": cond, "seed": seed, "outcome": outcome,
                   "genuine_success": inj.genuine_success(), "injected": inj.injected_count(),
                   "n_tool_calls": len(tools_used), "tools_used": tools_used,
                   "routed_correctly": bool(tools_used) and tools_used[0] == task["tool"],
                   "elapsed_s": round(time.time() - t0, 2), "events": inj.events,
                   "trajectory": traj, **lineage}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {task['id']:9s} {cond:12s} seed={seed} → {outcome}")


if __name__ == "__main__":
    main()
