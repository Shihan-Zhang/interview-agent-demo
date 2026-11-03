# 评测数据集

| 文件 | 来源 | 协议 | 内容 |
|---|---|---|---|
| `bfcl_subset_v1.jsonl` | [BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)(UC Berkeley Gorilla) | Apache-2.0 | 分层抽样 40 题:simple_python 15、multiple 10、parallel 5、irrelevance 10;种子 20261103;生成脚本 `eval/fc/build_bfcl_subset.py` |

字段:`id`、`source`、`license`、`category`、`messages`、`tools`(已转为标准 JSON Schema,函数名中的点号换成下划线)、`name_map`(换回原名用)、`ground_truth`(BFCL 的可接受取值列表)。
