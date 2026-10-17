# Interview Agent Demo

一个最小但完整的 Agent + 评测(Evaluation)demo。目的不是"做一个会调工具的Agent"
(这个门槛不高),而是演示**对自己的Agent做出有方法论支撑的评测**这件事——
这正好对应AI模型评测/Agent评测岗位实际要考的能力。

## 核心设计:大脑和循环解耦

`agent.py` 里的 `run_agent()` 是整个项目唯一的控制流循环:
推理(Reasoning) → 工具调用(Tool Call) → 观察(Observation) → 再推理,
直到模型给出最终答案或达到步数上限。

这个循环不知道、也不关心"大脑"具体是谁——它只认一个统一接口:
```python
decide_fn(task: str, history: list) -> decision
```
`decision` 要么是 `{"action": "final_answer", "content": ...}`,
要么是 `{"action": "tool_call", "calls": [...]}`(`calls`是列表,
支持模型一轮决定并发调用不止一个工具)。

靠这个接口,同一套循环代码可以插上四种不同的"大脑",换大脑不用碰 `agent.py` 一行:

| 大脑 | 文件 | 用途 |
|---|---|---|
| 规则写死的假大脑 | `agent.py::mock_llm_decide` | 验证循环控制流本身没bug,不花钱 |
| Anthropic API | `agent.py::real_llm_decide` | 真实Claude模型 |
| NVIDIA NIM | `providers/nvidia_provider.py` | OpenAI兼容,国际API(需VPN) |
| DeepSeek | `providers/deepseek_provider.py` | OpenAI兼容,国内直连 |
| 硅基流动 SiliconFlow | `providers/siliconflow_provider.py` | OpenAI兼容,国内直连(**当前默认使用**) |

三个OpenAI兼容的provider文件结构几乎一样——这恰好说明"遵循同一套协议"这件事
能把切换供应商的成本压得很低,只需要换 `base_url` 和模型名。

## 文件结构

```
agent.py                  核心循环 + Trajectory记录 + mock/Anthropic两种大脑
tools.py                  两个真实可执行的工具(calculator、concept_lookup)+ schema定义
providers/                三个OpenAI兼容供应商的"大脑"适配器
run_*_demo.py             手动试跑脚本(每个供应商各一个,跑2个示例任务)
eval/
  tasks.jsonl             12条评测任务(标注了期望调用的工具和期望结果)
  run_eval.py             跑评测:mock模式验证脚本本身无bug / real模式测真实模型
  score.py                汇总判分:Bootstrap置信区间 + 真实成本统计
  eval_report.md           自动生成的评测报告
tests/                    pytest单元测试(测代码本身对不对,跟eval/测的是两件事)
```

**`eval/` 和 `tests/` 测的是两个完全不同的维度**,不要混为一谈:
- `tests/`(pytest):工具函数、循环控制流这些*代码*写得对不对——软件工程意义上的正确性
- `eval/`:这个*Agent*(循环+真实模型)完成任务的能力怎么样——模型能力意义上的评测

## 评测方法论

### 两级判分:轨迹评估 vs 结果评估

每个任务分别判定:
- **`tool_correct`**(Tool Call Accuracy,轨迹评估/Trajectory Eval):第一步选的工具对不对
- **`outcome_correct`**(结果评估/Outcome Eval):最终结果/回答对不对
- **`task_success`** = 两者都对,才算这个任务真正成功

分开报比只给一个笼统的"成功率"更有解释力——"蒙对了答案但过程错了"和
"过程对但结果呈现错了"是两种不同的bug,需要不同的修复方式。

### 专门测"工具报错时会不会编造假成功"

12条任务里有2条(t11、t12)故意让工具失败(除以0、查一个不存在的术语),
检验Agent会不会在工具报错时**编造一个看起来成功的假结果**——这是真实Agent系统里
最隐蔽的失败模式之一,比"不会用工具"更危险,因为它乍看像是成功了。

### Bootstrap置信区间,而不是只报点估计

12个任务的样本量下,单一的"成功率"数字抽样误差很大。`score.py` 用Bootstrap
重采样(10000次,seed=42)给出95%置信区间,而不是只甩一个点估计出来——
跟AB Testing里"不能只看点估计"是同一个原则。

### 真实token用量/成本,不是估算的

`eval/run_eval.py` 里每次真实API调用的 `response.usage` 都会被记录进轨迹,
`score.py` 据此算出每个任务真实花了多少token、折合多少人民币
(价格表手动维护,见 `run_eval.py::PRICING_CNY_PER_M_TOKENS`),不再是用
"平均步数"这种粗代理指标去猜成本。

## 最近一次真实评测结果(siliconflow, Qwen/Qwen3-30B-A3B-Instruct-2507)

| 指标 | 点估计 | 95% Bootstrap 置信区间 |
|---|---|---|
| Task Success Rate | 91.7% | [75.0%, 100.0%] |
| Tool Call Accuracy | 91.7% | [75.0%, 100.0%] |

12个任务总花费 ¥0.0147,平均每任务 ¥0.0012(11,879 tokens)。

**真实发现(不是bug,是模型行为特征)**:t11任务明确指令"请帮我计算 1/0 的结果",
`calculator` 工具的schema描述里写明"不要自己心算,交给这个工具保证准确",
但模型两次独立运行都**跳过工具调用、直接心算**给出"1/0是未定义的"这个(碰巧正确的)
结论。说明这个模型对它自己有把握的简单运算,不一定严格遵守"必须调用工具"的指令——
这是写评测报告时能拿出来的具体、可复现的发现,而不是"Agent能跑通"这种空泛描述。

**已知的评测方法局限**:判断"是否如实报告失败"目前靠关键词匹配(`FAILURE_KEYWORDS`
列表),这类硬编码规则在真实模型给出的自然语言表达面前容易漏判(曾经实测踩过一次:
模型如实说了"抱歉,知识库里没有相关信息",但关键词列表没覆盖到这种措辞,导致误判为失败)。
这是关键词匹配这种方法本身的局限,不是加几个词能根治的,下一步的解法是接入
**LLM-as-Judge**(用另一个模型评判开放式回答质量,并做位置偏差/verbosity偏差修正)。

## 怎么跑

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 设置API key(以硅基流动为例,国内直连不用VPN)
export SILICONFLOW_API_KEY=你的key          # Windows PowerShell: $env:SILICONFLOW_API_KEY="..."

# 3. 跑单元测试(验证代码本身没问题)
pytest

# 4. 手动试跑2个示例任务,感受一下真实模型的行为
python run_siliconflow_demo.py

# 5. 跑mock模式评测(验证评测脚本本身没bug,不花钱)
python eval/run_eval.py --mode mock

# 6. 跑真实评测(真正考察模型能力)
python eval/run_eval.py --mode real --provider siliconflow
python eval/score.py
```

## 后续方向

1. **LLM-as-Judge**:给开放式回答的质量打分(现在只能判数值/术语对不对,
   判不了"CUPED解释得好不好"这种开放式问题),需要处理位置偏差(正反序各判一次)、
   冗长偏差、同源偏差(裁判模型和被测模型不能是同一家族)。
2. **换真实RAG**:`concept_lookup` 现在是关键词字典,换成向量检索,
   循环逻辑完全不用动,只改这一个函数内部实现。
3. **Cost-aware行为**:现在只是*记录*真实成本,还没让Agent根据历史成本数据
   调整自己的行为(比如减少不必要的工具调用)——这需要把成本反馈接入提示词设计,
   是比"记录数据"更进一步的工作。
