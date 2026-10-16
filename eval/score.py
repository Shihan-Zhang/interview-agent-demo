"""
对 run_eval.py 产出的逐任务判分结果做统计汇总。

核心是这句话:12个任务里成功了N个,"成功率"这个点估计本身不该直接拿去用——
样本量这么小,抽样误差很大,所以用 Bootstrap 重采样给出置信区间,
这和你 AB Testing 里"不能只看点估计,要看区间"的原则是同一件事,
只是从"实验指标"换成了"Agent 任务成功率"。
"""

import json
import os
import random

N_BOOTSTRAP = 10000
random.seed(42)


def load_results(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bootstrap_ci(successes: list, n_bootstrap: int = N_BOOTSTRAP, alpha: float = 0.05):
    """
    successes: 每个任务的 0/1 成功标记列表。
    用重采样模拟"如果重新抽一批同样多的任务,成功率会落在什么区间"。
    """
    n = len(successes)
    means = []
    for _ in range(n_bootstrap):
        resample = [random.choice(successes) for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lower = means[int((alpha / 2) * n_bootstrap)]
    upper = means[int((1 - alpha / 2) * n_bootstrap)]
    return lower, upper


def main():
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.jsonl")
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_report.md")

    rows = load_results(results_path)
    n = len(rows)

    task_successes = [1 if r["task_success"] else 0 for r in rows]
    tool_successes = [1 if r["tool_correct"] else 0 for r in rows]

    task_success_rate = sum(task_successes) / n
    tool_accuracy = sum(tool_successes) / n

    task_ci = bootstrap_ci(task_successes)
    tool_ci = bootstrap_ci(tool_successes)

    # 平均每个任务的步数,作为"成本"的代理指标(真实API接入后换成token数/花费)
    avg_steps = sum(len(r["trajectory"]["steps"]) for r in rows) / n

    failed = [r for r in rows if not r["task_success"]]

    lines = []
    lines.append("# Agent 评测报告\n")
    lines.append(f"任务总数: {n}\n")
    lines.append("## 核心指标\n")
    lines.append("| 指标 | 点估计 | 95% Bootstrap 置信区间 |")
    lines.append("|---|---|---|")
    lines.append(f"| Task Success Rate(任务级,结果+过程都对) | {task_success_rate:.1%} | "
                  f"[{task_ci[0]:.1%}, {task_ci[1]:.1%}] |")
    lines.append(f"| Tool Call Accuracy(轨迹级,工具选对没) | {tool_accuracy:.1%} | "
                  f"[{tool_ci[0]:.1%}, {tool_ci[1]:.1%}] |")
    lines.append(f"\n平均每任务步数(成本代理指标): {avg_steps:.2f}\n")

    lines.append("## 方法论说明\n")
    lines.append("- **区分结果评估(Outcome Eval)和轨迹评估(Trajectory Eval)**:"
                  "Task Success Rate 要求工具选对 *and* 最终结果对,"
                  "Tool Call Accuracy 单独衡量过程是否正确——"
                  "两者分开报告是因为'结果蒙对了但过程错了'和'过程对但结果展示错了'"
                  "是两种不同的失败,需要不同的修复方式。")
    lines.append("- **为什么要 Bootstrap 置信区间而不是只报点估计**:"
                  "12个任务的点估计方差很大,不加区间会让人误以为这个数字很精确。")
    lines.append("- **专门测了工具报错场景(t11/t12)**:"
                  "检验 Agent 在工具失败时会不会'编造一个假成功'——"
                  "这是真实 Agent 系统里最隐蔽的失败模式之一,"
                  "比'不会用工具'更危险,因为它看起来像是成功了。")

    if failed:
        lines.append("\n## 失败任务明细\n")
        for r in failed:
            lines.append(f"- **{r['id']}**: {r['task']}")
            lines.append(f"  - tool_correct={r['tool_correct']}, outcome_correct={r['outcome_correct']}, "
                          f"detail={r['detail']}")

    report = "\n".join(lines) + "\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"报告已保存到 {report_path}")


if __name__ == "__main__":
    main()
