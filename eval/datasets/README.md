# 评测数据集

| 文件 | 来源 | 协议 | 内容 |
|---|---|---|---|
| `bfcl_subset_v1.jsonl` | [BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)(UC Berkeley Gorilla) | Apache-2.0 | 分层抽样 40 题:simple_python 15、multiple 10、parallel 5、irrelevance 10;种子 20261103;生成脚本 `eval/fc/build_bfcl_subset.py` |
| `bfcl_subset_v2.jsonl` | 同上 | Apache-2.0 | 150 题 = v1 原样(前 40 题,顺序不变)+ live_simple 30、live_multiple 30、live_parallel 5、live_parallel_multiple 5、parallel_multiple 10、live_irrelevance 25、live_relevance 5;种子 20261104;排除题目加函数定义超过 16,000 字符的题(保证 num_ctx=8192 内不截断) |

字段:`id`、`source`、`license`、`category`、`messages`、`tools`(已转为标准 JSON Schema,函数名中的点号换成下划线)、`name_map`(换回原名用)、`ground_truth`(BFCL 的可接受取值列表)。

质量保证:`tests/test_fc_grader.py::test_every_ground_truth_passes_its_own_grader` 把每道题的标准答案当作模型输出,要求判分器全部判对(自洽检查)。
