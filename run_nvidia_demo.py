"""
用 NVIDIA NIM 的免费API,跑一遍真实模型版本的Agent。

跑之前:
1. export NVIDIA_API_KEY=你的nvapi-开头的key  (Windows PowerShell用: $env:NVIDIA_API_KEY="你的key")
2. pip install openai
"""

import os

from openai import OpenAI

from agent import run_agent
from providers.nvidia_provider import make_nvidia_decide_fn

# z-ai/glm-5.3 是在 build.nvidia.com 目录页面上实际确认过的、标注支持 Tool Use 的纯文本模型。
# 注意这个字符串是从模型页面的示例代码里原样抄来的(glm-5.3,点号不是横杠)——
# NVIDIA目录会不断上下架模型,这个字符串以后失效的话,去 build.nvidia.com 重新找一个
# 标"Tool Use"标签的模型,把这一行换掉就行,不用动其他任何代码。
MODEL = "z-ai/glm-5.3"


def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "没有找到环境变量 NVIDIA_API_KEY。"
            "Windows PowerShell 里先运行: $env:NVIDIA_API_KEY=\"你的nvapi-key\""
        )

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )
    decide_fn = make_nvidia_decide_fn(client, MODEL)

    test_tasks = [
        "任务成功率是 0.95,连续执行 8 步,整体成功率是多少?",
        "帮我解释一下 CUPED 是什么,为什么能缩短实验周期?",
    ]

    os.makedirs("trajectories", exist_ok=True)
    for i, task in enumerate(test_tasks):
        trajectory = run_agent(task, decide_fn=decide_fn)
        trajectory.print_summary()
        path = f"trajectories/nvidia_trajectory_{i}.json"
        trajectory.save(path)
        print(f"轨迹已保存到 {path}")


if __name__ == "__main__":
    main()
