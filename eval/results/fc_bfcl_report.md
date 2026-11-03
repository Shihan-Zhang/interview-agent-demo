# BFCL 子集函数调用评测

BFCL v4 分层抽样 40 题(Apache-2.0)。每题每配置 1 次,准确率用 Wilson 95% 置信区间,比较用精确 McNemar 检验。

## 各配置结果

| 配置 | 有效 | 准确率 [95% CI] | simple_python | 每题耗时 | 每题输出 token |
|---|---|---|---|---|---|
| qwen3-think | 6/6 | 0/6 = 0.0% [0.0%, 39.0%] | 0/6 | 0.0 s | 30 |
| qwen3i-nothink | 6/6 | 0/6 = 0.0% [0.0%, 39.0%] | 0/6 | 0.0 s | 30 |
| qwen35-think | 6/6 | 0/6 = 0.0% [0.0%, 39.0%] | 0/6 | 0.0 s | 30 |
| qwen35-nothink | 6/6 | 0/6 = 0.0% [0.0%, 39.0%] | 0/6 | 0.0 s | 30 |

## 配对比较(精确 McNemar)

| 比较 | 都对 | 只有 A 对 (b) | 只有 B 对 (c) | 都错 | 准确率差 | p 值 | 说明 |
|---|---|---|---|---|---|---|---|
| qwen35-nothink vs qwen3i-nothink | 0 | 0 | 0 | 6 | +0.0% | 1.000 | **主要比较** |
| qwen35-think vs qwen35-nothink | 0 | 0 | 0 | 6 | +0.0% | 1.000 | 探索性 |
| qwen3-think vs qwen3i-nothink | 0 | 0 | 0 | 6 | +0.0% | 1.000 | 探索性 |
| qwen35-think vs qwen3-think | 0 | 0 | 0 | 6 | +0.0% | 1.000 | 探索性 |

## 失败明细(用于 Trace 分析)

| 配置 | 题 | 类别 | 原因 |
|---|---|---|---|
| qwen3-think | simple_python_130 | simple_python | 调用次数 0 ≠ 期望 1 |
| qwen3-think | simple_python_260 | simple_python | 缺少必需参数:area |
| qwen3-think | simple_python_27 | simple_python | 缺少必需参数:initial_velocity |
| qwen3-think | simple_python_270 | simple_python | 缺少必需参数:building_name |
| qwen3-think | simple_python_271 | simple_python | 缺少必需参数:building_id |
| qwen3-think | simple_python_315 | simple_python | 缺少必需参数:season |
| qwen3i-nothink | simple_python_130 | simple_python | 缺少必需参数:cash_flows |
| qwen3i-nothink | simple_python_260 | simple_python | 调用次数 0 ≠ 期望 1 |
| qwen3i-nothink | simple_python_27 | simple_python | 缺少必需参数:initial_velocity |
| qwen3i-nothink | simple_python_270 | simple_python | 缺少必需参数:building_name |
| qwen3i-nothink | simple_python_271 | simple_python | 调用次数 0 ≠ 期望 1 |
| qwen3i-nothink | simple_python_315 | simple_python | 缺少必需参数:season |
| qwen35-think | simple_python_130 | simple_python | 缺少必需参数:cash_flows |
| qwen35-think | simple_python_260 | simple_python | 缺少必需参数:area |
| qwen35-think | simple_python_27 | simple_python | 缺少必需参数:initial_velocity |
| qwen35-think | simple_python_270 | simple_python | 缺少必需参数:building_name |
| qwen35-think | simple_python_271 | simple_python | 缺少必需参数:building_id |
| qwen35-think | simple_python_315 | simple_python | 缺少必需参数:season |
| qwen35-nothink | simple_python_130 | simple_python | 缺少必需参数:cash_flows |
| qwen35-nothink | simple_python_260 | simple_python | 缺少必需参数:area |
| qwen35-nothink | simple_python_27 | simple_python | 缺少必需参数:initial_velocity |
| qwen35-nothink | simple_python_270 | simple_python | 缺少必需参数:building_name |
| qwen35-nothink | simple_python_271 | simple_python | 缺少必需参数:building_id |
| qwen35-nothink | simple_python_315 | simple_python | 调用次数 0 ≠ 期望 1 |
