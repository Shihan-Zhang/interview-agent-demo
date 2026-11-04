"""
本地模型对比实验:2 个模型 × 思考模式开/关 = 4 个配置,每个配置把 12 道题各跑 3 次。

设计(详见 README 的"本地模型对比"一节):
- 采样参数用各模型的推荐设置(方案 A),每个配置的参数都记录在结果里;
- 同一道题、同一次运行,在所有配置下用同一个随机种子(公共随机数,配对比较更灵敏);
- 每个配置开始前先卸载所有模型、测显存基线,再加载、再测,避免后台程序干扰;
- 结果逐条追加写入 JSONL,中途中断后重新运行会自动跳过已完成的部分。

运行(在项目根目录):
    python eval/compare_local.py              # 跑全部 4 个配置
    python eval/compare_local.py --runs 1     # 先每题只跑 1 次,快速试跑
    python eval/summarize_local.py            # 跑完后生成报告
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from agent import run_agent  # noqa: E402
from providers.ollama_provider import DEFAULT_BASE_URL, loaded_models, make_ollama_decide_fn  # noqa: E402
from run_eval import check_outcome, load_tasks  # noqa: E402

COMMON = {"num_ctx": 4096, "num_predict": 2048}   # num_predict 防止思考模式停不下来

# 方案 A:各模型的推荐采样参数
# - Qwen3:官方说明里思考 / 非思考模式分别推荐的参数
# - Qwen3.5:用 Ollama 模型文件里的默认值(ollama show 看到的那组)
CONFIGS = [
    {"name": "qwen3-think",    "model": "qwen3:4b",   "think": True,
     "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}},
    {"name": "qwen3-nothink",  "model": "qwen3:4b",   "think": False,
     "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}},
    # 修正版 v2:think=False 加 /no_think 软开关,仍然 12/12 泄漏(操纵检查不通过)。
    {"name": "qwen3-nothink-v2", "model": "qwen3:4b", "think": False, "user_suffix": " /no_think",
     "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}},
    # 根因:Ollama 的 qwen3:4b 是 Qwen3-4B-Thinking-2507(只有思考模式,上下文 262144,
    # ollama show 里 thinking levels 只有 true),开关对它无效。
    # 不思考的对照应该用另一个模型文件 Qwen3-4B-Instruct-2507。
    {"name": "qwen3i-nothink", "model": "qwen3:4b-instruct", "think": False,
     "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}},
    {"name": "qwen35-think",   "model": "qwen3.5:4b", "think": True,
     "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}},
    {"name": "qwen35-nothink", "model": "qwen3.5:4b", "think": False,
     "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}},
]

RESULTS = os.path.join(ROOT, "eval", "results", "local_compare.jsonl")
META = os.path.join(ROOT, "eval", "results", "local_compare_meta.jsonl")


def nvidia_used_mib():
    """nvidia-smi 读当前显存占用(MiB);读不到就返回 None,不影响实验。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def unload_all():
    """把 Ollama 里已加载的模型全部卸载(keep_alive=0),让显存回到基线。"""
    for m in loaded_models():
        body = json.dumps({"model": m["name"], "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(f"{DEFAULT_BASE_URL}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
    time.sleep(3)


def warm_up(model: str, think: bool):
    """加载模型并跑一个很短的请求,让后面计时不包含加载时间。"""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "你好"}],
                       "think": think, "stream": False, "keep_alive": "30m",
                       "options": {**COMMON, "num_predict": 16}}).encode("utf-8")
    req = urllib.request.Request(f"{DEFAULT_BASE_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8")).get("load_duration", 0) / 1e9


def done_keys():
    keys = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") == "ok":
                    keys.add((r["config"], r["id"], r["run"]))
    return keys


def sum_usage(traj: dict) -> dict:
    keys = ["prompt_tokens", "completion_tokens", "prefill_s", "decode_s", "total_s", "thinking_chars"]
    tot = {k: 0.0 for k in keys}
    calls, truncated, first_prefill = 0, False, None
    for s in traj["steps"]:
        u = s["content"].get("usage") if isinstance(s["content"], dict) else None
        if u:
            calls += 1
            for k in keys:
                tot[k] += u.get(k) or 0
            truncated = truncated or u.get("done_reason") == "length"
            if first_prefill is None:
                first_prefill = u.get("prefill_s")
    tot.update({"llm_calls": calls, "truncated": truncated, "first_prefill_s": first_prefill})
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="每道题每个配置跑几次")
    ap.add_argument("--configs", default="all", help="逗号分隔的配置名,默认全部")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

    tasks = load_tasks(os.path.join(ROOT, "eval", "tasks.jsonl"))
    configs = CONFIGS if args.configs == "all" else [c for c in CONFIGS if c["name"] in args.configs.split(",")]
    names = {c["name"] for c in configs}
    finished = {k for k in done_keys() if k[0] in names and k[2] <= args.runs}
    total = len(configs) * args.runs * len(tasks)
    print(f"共 {total} 次运行;已完成 {len(finished)} 次,会自动跳过。")
    t_start, n_done_now = time.time(), 0

    for cfg in configs:
        todo = [(run, i, t) for run in range(1, args.runs + 1) for i, t in enumerate(tasks)
                if (cfg["name"], t["id"], run) not in finished]
        if not todo:
            print(f"\n[{cfg['name']}] 已全部完成,跳过")
            continue

        # ---- 显存:先卸载所有模型测基线,再加载测占用 ----
        unload_all()
        base = nvidia_used_mib()
        load_s = warm_up(cfg["model"], cfg["think"])
        after = nvidia_used_mib()
        ps = [m for m in loaded_models() if m["name"].startswith(cfg["model"])]
        meta = {"config": cfg["name"], "model": cfg["model"], "vram_base_mib": base, "vram_after_mib": after,
                "vram_delta_mib": (after - base) if (base is not None and after is not None) else None,
                "ollama_size": ps[0].get("size") if ps else None,
                "ollama_size_vram": ps[0].get("size_vram") if ps else None,
                "load_s": load_s, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(META, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        gpu_share = (meta["ollama_size_vram"] / meta["ollama_size"]) if ps and meta["ollama_size"] else None
        print(f"\n[{cfg['name']}] 显存 基线 {base} MiB -> 加载后 {after} MiB;"
              f"Ollama 报告 {meta['ollama_size']} 字节,GPU 占比 {gpu_share}")

        for run, i, task in todo:
            options = {**COMMON, **cfg["sampling"], "seed": run * 100 + i}   # 各配置同题同次用同一种子
            decide_fn = make_ollama_decide_fn(cfg["model"], cfg["think"], options,
                                              user_suffix=cfg.get("user_suffix", ""))
            row = {"config": cfg["name"], "model": cfg["model"], "think": cfg["think"],
                   "options": options, "user_suffix": cfg.get("user_suffix", ""), "id": task["id"], "run": run, "check_type": task["check_type"]}
            t0 = time.time()
            try:
                traj = run_agent(task["task"], decide_fn=decide_fn, verbose=False)
                row.update(check_outcome(task, traj))
                row.update({"status": "ok", "usage": sum_usage(traj.to_dict()), "trajectory": traj.to_dict()})
            except Exception as e:                       # 单条出错不中断整个实验,记下来继续
                row.update({"status": "error", "error": repr(e), "task_success": False})
            row["wall_s"] = time.time() - t0

            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_done_now += 1
            avg = (time.time() - t_start) / n_done_now
            remaining = total - len(finished) - n_done_now
            flag = "PASS" if row.get("task_success") else ("ERROR" if row["status"] == "error" else "FAIL")
            print(f"  [{cfg['name']}] run{run} {task['id']} {flag:5s} {row['wall_s']:6.1f}s"
                  f"  | 剩余约 {remaining} 次,预计 {remaining * avg / 60:.0f} 分钟")

    print(f"\n全部完成,结果在 {RESULTS}。运行 python eval/summarize_local.py 生成报告。")


if __name__ == "__main__":
    main()
