# BFCL 子集函数调用评测(bfcl_v2)

数据集 `bfcl_subset_v2.jsonl`,150 题,BFCL v4 分层抽样(Apache-2.0)。每题每配置 1 次,准确率用 Wilson 95% 置信区间,比较用精确 McNemar 检验。

## 数据质量检查

- 重新判分改变的行:5
- 来源可疑(耗时 < 0.05 s):0
- 输入可能被截断(≥ 95% num_ctx):0
- 输出被截断(done_reason = length,一律判失败):4

## 各配置总体结果

| 配置 | 完成题数 | 有效 | 准确率 [95% CI] | 每题耗时 | 每题输出 token |
|---|---|---|---|---|---|
| qwen3-think | 40/150 | 40/40 | 37/40 = 92.5% [80.1%, 97.4%] | 11.2 s | 746 |
| qwen3i-nothink | 150/150 | 150/150 | 126/150 = 84.0% [77.3%, 89.0%] | 3.3 s | 72 |
| qwen35-think | 40/150 | 40/40 | 35/40 = 87.5% [73.9%, 94.5%] | 6.3 s | 288 |
| qwen35-nothink | 150/150 | 150/150 | 119/150 = 79.3% [72.2%, 85.0%] | 3.8 s | 95 |

## 分类别准确率

| 类别 | 题数 | qwen3-think | qwen3i-nothink | qwen35-think | qwen35-nothink |
|---|---|---|---|---|---|
| simple_python | 15 | 14/15 | 14/15 | 14/15 | 14/15 |
| multiple | 10 | 10/10 | 9/10 | 8/10 | 10/10 |
| parallel | 5 | 5/5 | 5/5 | 4/5 | 4/5 |
| irrelevance | 10 | 8/10 | 10/10 | 9/10 | 9/10 |
| live_simple | 30 | — | 25/30 | — | 20/30 |
| live_multiple | 30 | — | 19/30 | — | 21/30 |
| live_parallel | 5 | — | 3/5 | — | 2/5 |
| live_parallel_multiple | 5 | — | 4/5 | — | 3/5 |
| parallel_multiple | 10 | — | 10/10 | — | 10/10 |
| live_irrelevance | 25 | — | 23/25 | — | 21/25 |
| live_relevance | 5 | — | 4/5 | — | 5/5 |

## 配对比较(精确 McNemar)

| 比较 | 共同题数 | 都对 | 只有 A 对 (b) | 只有 B 对 (c) | 都错 | 准确率差 | p 值 | 说明 |
|---|---|---|---|---|---|---|---|---|
| qwen35-nothink vs qwen3i-nothink | 150 | 110 | 9 | 16 | 15 | -4.7% | 0.230 | **主要比较** |
| qwen35-think vs qwen35-nothink | 40 | 32 | 3 | 5 | 0 | -5.0% | 0.727 | 探索性 |
| qwen3-think vs qwen3i-nothink | 40 | 35 | 2 | 3 | 0 | -2.5% | 1.000 | 探索性 |
| qwen35-think vs qwen3-think | 40 | 33 | 2 | 4 | 1 | -5.0% | 0.688 | 探索性 |

## 失败明细(用于 Trace 分析)

| 配置 | 题 | 类别 | 原因 |
|---|---|---|---|
| qwen3-think | irrelevance_18 | irrelevance | 输出截断:2048 token 内未完成(思考 8827 字符,调用 0 次) |
| qwen3-think | irrelevance_30 | irrelevance | 输出截断:2048 token 内未完成(思考 9611 字符,调用 0 次) |
| qwen3-think | simple_python_130 | simple_python | 输出截断:2048 token 内未完成(思考 7422 字符,调用 0 次) |
| qwen3i-nothink | live_irrelevance_318-76-16 | live_irrelevance | 不应调用函数,却调用了 requests.get |
| qwen3i-nothink | live_irrelevance_81-2-69 | live_irrelevance | 输出截断:2048 token 内未完成(思考 0 字符,调用 43 次) |
| qwen3i-nothink | live_multiple_1012-241-0 | live_multiple | 参数 filter 的值 'name:alpha,beta,gamma' 不在可接受范围 ['details:alpha,beta,gamma'] |
| qwen3i-nothink | live_multiple_1013-242-0 | live_multiple | 参数 filter 的值 'name:alpha OR name:beta OR name:gamma' 不在可接受范围 [''] |
| qwen3i-nothink | live_multiple_1033-261-0 | live_multiple | 参数 filter 的值 "name in ['alpha', 'beta', 'gamma']" 不在可接受范围 ["name IN ('alpha', 'beta', 'gamma')", 'name:alpha OR name:bet |
| qwen3i-nothink | live_multiple_24-5-1 | live_multiple | 参数 color 的值 'shimmer' 不在可接受范围 ['', None] |
| qwen3i-nothink | live_multiple_433-141-22 | live_multiple | 参数 theater_name 的值 'Century Napa Valley' 不在可接受范围 ['Century Napa Valley and XD'] |
| qwen3i-nothink | live_multiple_506-149-3 | live_multiple | 缺少必需参数:number_of_tickets |
| qwen3i-nothink | live_multiple_597-158-3 | live_multiple | 参数 number_of_rooms 的值 '1' 不在可接受范围 ['', 'dontcare'] |
| qwen3i-nothink | live_multiple_642-161-10 | live_multiple | 缺少必需参数:artist |
| qwen3i-nothink | live_multiple_720-165-7 | live_multiple | 参数 rating 的值 4.5 不在可接受范围 ['', 3.0] |
| qwen3i-nothink | live_multiple_853-180-1 | live_multiple | 函数名错误:Buses_3_FindBus ≠ Buses_3_BuyBusTicket |
| qwen3i-nothink | live_multiple_926-191-14 | live_multiple | 缺少必需参数:start_available_date |
| qwen3i-nothink | live_parallel_15-11-0 | live_parallel | 并行调用与标准答案无法一一对应:多传了参数:['unit']; 参数 command 的值 'echo hello > c:\\testing.txt' 不在可接受范围 ['echo.>C:\\testing.txt', 'echo.>C: |
| qwen3i-nothink | live_parallel_3-0-3 | live_parallel | 并行调用与标准答案无法一一对应:参数 unit 的值 'celsius' 不在可接受范围 ['', 'fahrenheit']; 参数 unit 的值 'celsius' 不在可接受范围 ['', 'fahrenheit']; 参数 uni |
| qwen3i-nothink | live_parallel_multiple_13-11-0 | live_parallel_multiple | 并行调用与标准答案无法一一对应:ok; 参数 function 的值 'x ** 2' 不在可接受范围 ['lambda x: x**2'] |
| qwen3i-nothink | live_relevance_6-6-0 | live_relevance | 应调用函数,却没有调用 |
| qwen3i-nothink | live_simple_107-64-0 | live_simple | 参数 faq_describe_accounts_start 的值 ["I'm interested in opening a long term savings account. What options do you have?", ' |
| qwen3i-nothink | live_simple_158-95-15 | live_simple | 参数 unit 的值 'seconds' 不在可接受范围 ['', 'N/A'] |
| qwen3i-nothink | live_simple_165-98-0 | live_simple | 调用次数 0 ≠ 期望 1 |
| qwen3i-nothink | live_simple_18-3-14 | live_simple | 参数 location 的值 'Moscow, RU' 不在可接受范围 ['Moscow, Russia'] |
| qwen3i-nothink | live_simple_99-59-0 | live_simple | 参数 auto_reconnect 的值 True 不在可接受范围 ['', False] |
| qwen3i-nothink | multiple_80 | multiple | 缺少必需参数:instruments |
| qwen3i-nothink | simple_python_99 | simple_python | 参数 end_range 的值 2 不在可接受范围 [6.2832] |
| qwen35-think | irrelevance_123 | irrelevance | 不应调用函数,却调用了 calculate_bmi |
| qwen35-think | multiple_164 | multiple | 参数 finish 的值 'Rosewood Finish' 不在可接受范围 ['Rosewood'] |
| qwen35-think | multiple_80 | multiple | 参数 services 的值 ['Guitar Lessons', 'Violin Lessons'] 不在可接受范围 [['Violin Lessons']] |
| qwen35-think | parallel_116 | parallel | 调用次数 1 ≠ 期望 3 |
| qwen35-think | simple_python_130 | simple_python | 参数 discount_rate 的值 8 不在可接受范围 [0.08] |
| qwen35-nothink | irrelevance_133 | irrelevance | 不应调用函数,却调用了 get_battle_details |
| qwen35-nothink | live_irrelevance_104-4-0 | live_irrelevance | 不应调用函数,却调用了 ChaFod |
| qwen35-nothink | live_irrelevance_170-24-0 | live_irrelevance | 不应调用函数,却调用了 todo |
| qwen35-nothink | live_irrelevance_318-76-16 | live_irrelevance | 不应调用函数,却调用了 requests.get |
| qwen35-nothink | live_irrelevance_781-290-0 | live_irrelevance | 不应调用函数,却调用了 EventSettingsApi.get_event_specification_infos |
| qwen35-nothink | live_multiple_1010-239-0 | live_multiple | 参数 query 的值 'name=dashing' 不在可接受范围 ['dashing'] |
| qwen35-nothink | live_multiple_1012-241-0 | live_multiple | 参数 filter 的值 'name:alpha OR name:beta OR name:gamma' 不在可接受范围 ['details:alpha,beta,gamma'] |
| qwen35-nothink | live_multiple_1033-261-0 | live_multiple | 参数 filter 的值 'name=alpha OR name=beta OR name=gamma' 不在可接受范围 ["name IN ('alpha', 'beta', 'gamma')", 'name:alpha OR name: |
| qwen35-nothink | live_multiple_24-5-1 | live_multiple | 参数 color 的值 'shimmer' 不在可接受范围 ['', None] |
| qwen35-nothink | live_multiple_433-141-22 | live_multiple | 参数 theater_name 的值 'Century Napa Valley' 不在可接受范围 ['Century Napa Valley and XD'] |
| qwen35-nothink | live_multiple_506-149-3 | live_multiple | 缺少必需参数:number_of_tickets |
| qwen35-nothink | live_multiple_597-158-3 | live_multiple | 参数 number_of_rooms 的值 '1' 不在可接受范围 ['', 'dontcare'] |
| qwen35-nothink | live_multiple_647-161-15 | live_multiple | 多传了参数:['dontcare'] |
| qwen35-nothink | live_multiple_926-191-14 | live_multiple | 缺少必需参数:province_id |
| qwen35-nothink | live_parallel_15-11-0 | live_parallel | 调用次数 1 ≠ 期望 2 |
| qwen35-nothink | live_parallel_3-0-3 | live_parallel | 并行调用与标准答案无法一一对应:参数 location 的值 'Cancún, Mexico' 不在可接受范围 ['Cancún, QR', 'Canc\\u00fan, QR']; 参数 location 的值 'Playa del Ca |
| qwen35-nothink | live_parallel_4-1-0 | live_parallel | 并行调用与标准答案无法一一对应:参数 url 的值 'https://api.open-meteo.com/v1/forecast?latitude=42.3601&longitude=71.0589&current=true' 不在可接受 |
| qwen35-nothink | live_parallel_multiple_13-11-0 | live_parallel_multiple | 调用次数 1 ≠ 期望 2 |
| qwen35-nothink | live_parallel_multiple_21-18-0 | live_parallel_multiple | 并行调用与标准答案无法一一对应:参数 is_unisex 的值 False 不在可接受范围 ['', 'dontcare']; ok |
| qwen35-nothink | live_simple_107-64-0 | live_simple | 参数 faq_describe_accounts_start 的值 ["I'm interested in opening a long term savings account. What options do you have?", ' |
| qwen35-nothink | live_simple_114-70-0 | live_simple | 参数 notify 的值 False 不在可接受范围 ['', True] |
| qwen35-nothink | live_simple_117-73-0 | live_simple | 参数 input_value 的值 'hi' 不在可接受范围 ['say hi'] |
| qwen35-nothink | live_simple_120-76-0 | live_simple | 参数 include_wind 的值 True 不在可接受范围 ['', False] |
| qwen35-nothink | live_simple_138-91-0 | live_simple | 参数 dateOrTime 的值 '2023-11-01T16:00:00Z' 不在可接受范围 ['2023-11-01T16:00:00'] |
| qwen35-nothink | live_simple_165-98-0 | live_simple | 参数 data 的值 [{'age': 18, 'name': '李雷'}] 不在可接受范围 [[{'name': ['李雷'], 'age': [18]}, {'name': ['李丽'], 'age': [21]}]] |
| qwen35-nothink | live_simple_177-103-0 | live_simple | 调用次数 0 ≠ 期望 1 |
| qwen35-nothink | live_simple_18-3-14 | live_simple | 参数 location 的值 'Moscow, RU' 不在可接受范围 ['Moscow, Russia'] |
| qwen35-nothink | live_simple_189-114-0 | live_simple | 参数 data 的值 [{'age': 43, 'name': 'Jane', 'nick_name': None}] 不在可接受范围 [[{'name': ['Chester'], 'age': [42]}, {'name': ['Jan |
| qwen35-nothink | live_simple_7-3-3 | live_simple | 参数 location 的值 'Riga, LV' 不在可接受范围 ['Riga, Latvia'] |
| qwen35-nothink | parallel_52 | parallel | 并行调用与标准答案无法一一对应:参数 preferences 的值 ['Vegetarian', 'Delivery'] 不在可接受范围 [['Vegetarian']]; ok |
| qwen35-nothink | simple_python_99 | simple_python | 参数 end_range 的值 6.2831 不在可接受范围 [6.2832] |
