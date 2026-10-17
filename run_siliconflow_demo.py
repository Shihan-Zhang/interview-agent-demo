"""
用硅基流动(SiliconFlow)的API,跑一遍真实模型版本的Agent。
国内直连,不需要VPN。

跑之前:
1. 去 siliconflow.cn 注册、生成API key
2. Windows PowerShell: $env:SILICONFLOW_API_KEY="你的key"
3. pip install openai  (之前装过的话这一步可以跳过)
"""

import os

from openai import OpenAI

from agent import run_agent
from providers.siliconflow_provider import make_siliconflow_decide_fn

# Qwen/Qwen3-30B-A3B-Instruct-2507: 在用户实际看到的硅基流动模型目录页上确认
# 标有"Tools"标签(支持工具调用)。选Instruct版而不是默认开思考模式的版本,
# 是因为我们这个场景只需要它干脆地选工具、吐参数,不需要大段思考过程——
# 更快、更省token、也更利于稳定解析。MoE架构,只激活3B参数,速度快。
# (注:之前用的 Qwen3-8B 已经在该目录页上找不到了,说明已下架——
#  供应商模型目录会随时变动,这个常量以后失效的话,去硅基流动官网模型页
#  重新挑一个带"Tools"标签的,把这一行换掉就行,不用动其他代码。)
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def main():
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "没有找到环境变量 SILICONFLOW_API_KEY。"
            "Windows PowerShell 里先运行: $env:SILICONFLOW_API_KEY=\"你的key\""
        )

    client = OpenAI(
        base_url="https://api.siliconflow.cn/v1",
        api_key=api_key,
    )
    decide_fn = make_siliconflow_decide_fn(client, MODEL)

    test_tasks = [
        "任务成功率是 0.95,连续执行 8 步,整体成功率是多少?",
        "帮我解释一下 CUPED 是什么,为什么能缩短实验周期?",
    ]

    os.makedirs("trajectories", exist_ok=True)
    for i, task in enumerate(test_tasks):
        trajectory = run_agent(task, decide_fn=decide_fn)
        trajectory.print_summary()
        path = f"trajectories/siliconflow_trajectory_{i}.json"
        trajectory.save(path)
        print(f"轨迹已保存到 {path}")


if __name__ == "__main__":
    main()
