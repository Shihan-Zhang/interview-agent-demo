"""
生成故障注入评测用的固定数据:虚构岗位表 + 任务集。运行:python eval/agent_faults/build_fixtures.py

为什么用虚构公司:模型不可能凭记忆答对,"答对"就说明真的用上了工具返回的数据;
否则"工具宕机时答对"分不清是恢复了还是蒙对了。
标准答案用这里的列表推导独立计算,不调用 router/jobs.py——用被测代码算标准答案是循环论证。
"""
import json, os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY = "2026-11-05"

def J(company, title, city, status, tracks, rtype, prio, applied, nxt):
    return {"company": company, "title": title, "city": city, "status": status, "tracks": tracks,
            "recruit_type": rtype, "priority": prio, "applied_date": applied, "next_date": nxt,
            "jd": f"{title}:负责相关业务的数据分析与建模。"}

JOBS = [
    J("星衡科技", "大模型评测工程师", "上海", "已投递", ["互联网数据分析"], "校招", "高", "2026-10-08", None),
    J("岚舟智能", "Agent 评测算法工程师", "上海", "一面", ["互联网数据分析"], "校招", "高", "2026-10-12", "2026-11-12"),
    J("青檀医疗", "医疗数据科学家", "上海", "待投递", ["医疗AI/健康数据"], "校招", None, None, "2026-11-20"),
    J("禾川健康", "真实世界数据分析师", "北京", "待投递", ["医疗AI/健康数据", "药企/RWE"], "社招", "中", None, None),
    J("澄宇医学", "医学影像算法实习生", "上海", "已投递", ["医疗AI/健康数据"], "实习", "中", "2026-10-21", None),
    J("北辰电商", "风控策略分析师", "杭州", "笔试/测评", ["电商风控"], "校招", "高", "2026-10-15", "2026-11-09"),
    J("云杉金融", "信贷风控建模", "深圳", "已拒/挂", ["金融风控"], "校招", "中", "2026-10-03", None),
    J("曜石数据", "数据分析师", "上海", "已投递", ["互联网数据分析"], "校招", "中", "2026-10-25", None),
    J("霁光AI", "大模型数据工程师", "北京", "二面", ["其他"], "校招", "高", "2026-10-10", "2026-11-14"),
    J("松岩游戏", "AI 应用开发实习生", "上海", "已投递", ["其他"], "实习", "低", "2026-10-28", None),
    J("鹭汀咨询", "商业分析", "上海", "放弃", ["互联网数据分析"], "校招", "低", "2026-10-05", None),
    J("知微生物", "生物统计分析师", "上海", "待投递", ["药企/RWE"], "校招", "中", None, "2026-11-30"),
    J("远帆出行", "数据科学工程师", "上海", "已投递", ["互联网数据分析"], "校招", "高", "2026-10-11", None),
    J("玄鸟安全", "模型评测与红队实习生", "北京", "已投递", ["其他"], "实习", "中", "2026-11-02", None),
    J("苍梧零售", "数据分析实习生", "其他", "已投递", ["电商风控"], "实习", "低", "2026-10-30", None),
    J("沐川保险", "精算数据分析", "深圳", "待投递", ["金融风控"], "社招", "低", None, None),
]

notes_tasks = [  # 复用 eval/rag 的题(答案在 Shihan 的笔记里,模型不可能凭记忆知道这些具体数字)
    ("notes-01", "m5-01", "想把评测准确率的置信区间控制在正负 5 个百分点,大概要多少道题?", [r"246"], "约 246 道题(p = 0.8 时)"),
    ("notes-02", "m5-02", "McNemar 检验里 b=6、c=1 时 p 值是多少?", [r"0\.125"], "p = 0.125"),
    ("notes-03", "m5-05", "我的笔记里说要以足够功效测出 5 个百分点的差距,大概需要多少题?", [r"480"], "约 480 题"),
    ("notes-04", "m5-09", "为什么 qwen3 关思考的对照组换成了 Instruct 版?笔记里记录的失败次数是多少?", [r"36\s*/\s*36|36\s*次"], "关思考 36/36 次失败,Qwen3-4B 是只会思考的版本"),
    ("notes-05", "m5-14", "按我的评测规则,输出被截断的题应该怎么算分?", [r"(判|算|记)(为)?(失败|错|0)"], "截断一律判失败"),
    ("notes-06", "plan-02", "我的学习计划里 Agent 评测集计划建多少条任务?", [r"50"], "约 50 条任务"),
    ("notes-07", "m1-04", "笔记里说用了 GQA(8 组)以后 KV cache 降到多少?", [r"0\.5\s*GB"], "降到 0.5 GB"),
    ("notes-08", "m5-10", "思考模式要不要默认打开?笔记里的结论是什么?", [r"(默认)?关闭|不(要|用)(默认)?(打开|开启)"], "耗时 2–3 倍,默认关闭"),
]

COUNTS = {}

def jobs_tasks():
    t = []
    n = sum(1 for j in JOBS if j["city"] == "上海" and j["status"] == "已投递")
    COUNTS["jobs-01"] = n
    t.append(("jobs-01", "状态是「已投递」的上海岗位一共有几个?", [rf"(?<!\d){n}(?!\d)\s*(个|家|条)"]))
    cos = [j["company"] for j in JOBS if "医疗AI/健康数据" in j["tracks"] and j["status"] == "待投递"]
    t.append(("jobs-02", "医疗AI/健康数据赛道里还没投递(待投递)的岗位是哪几家公司?", cos))
    cos = [j["company"] for j in JOBS if j["next_date"] and j["next_date"] <= "2026-11-15"]
    t.append(("jobs-03", "下次节点(笔试/面试/截止)在 11 月 15 日及之前的岗位有哪些公司?", cos))
    n = sum(1 for j in JOBS if j["applied_date"] and j["applied_date"] >= "2026-10-26")
    COUNTS["jobs-04"] = n
    t.append(("jobs-04", "10 月 26 日(含)之后投递的岗位有几个?", [rf"(?<!\d){n}(?!\d)\s*(个|家|条)"]))
    cos = [j["company"] for j in JOBS if j["priority"] == "高" and j["status"] == "二面"]
    t.append(("jobs-05", "优先级为高、状态是二面的岗位是哪家公司?", cos))
    first = min((j for j in JOBS if j["applied_date"]), key=lambda j: j["applied_date"])
    t.append(("jobs-06", "我投递日期最早的岗位是哪家公司?", [first["company"]]))
    n = sum(1 for j in JOBS if j["recruit_type"] == "实习")
    COUNTS["jobs-07"] = n
    t.append(("jobs-07", "岗位表里实习岗位一共有几个?", [rf"(?<!\d){n}(?!\d)\s*(个|家|条)"]))
    cos = [j["company"] for j in JOBS if "评测" in j["title"]]
    t.append(("jobs-08", "岗位名称里带“评测”的有哪几家公司?", cos))
    return t

# 虚构新闻:标注 synthetic,避免被当成真实信息;同样保证模型不能凭记忆答对
WEB = [
    ("web-01", "星衡科技最近发布的大模型评测岗位要求几年工作经验?",
     [{"title": "星衡科技发布 2027 届大模型评测岗位", "url": "https://example.com/xh-jobs", "published": "2026-11-03",
       "snippet": "星衡科技评测团队招聘大模型评测工程师,要求 1 年以上相关实习或工作经验,熟悉 Python。"}], [r"1\s*年"]),
    ("web-02", "岚舟智能的 Agent 评测团队上周公布的基准叫什么名字?",
     [{"title": "岚舟智能开源 Agent 评测基准 LZ-Bench", "url": "https://example.com/lz-bench", "published": "2026-10-30",
       "snippet": "岚舟智能发布 LZ-Bench,包含 820 道多轮工具调用任务。"}], [r"LZ-?Bench"]),
    ("web-03", "LZ-Bench 一共包含多少道任务?",
     [{"title": "岚舟智能开源 Agent 评测基准 LZ-Bench", "url": "https://example.com/lz-bench", "published": "2026-10-30",
       "snippet": "岚舟智能发布 LZ-Bench,包含 820 道多轮工具调用任务,覆盖 12 个领域。"}], [r"820"]),
    ("web-04", "霁光AI 的 2027 届秋招网申截止日期是哪天?",
     [{"title": "霁光AI 2027 届秋季校园招聘启动", "url": "https://example.com/jg-campus", "published": "2026-09-20",
       "snippet": "网申截止时间为 2026 年 11 月 25 日,笔试安排在 12 月初。"}], [r"11\s*月\s*25|11-25"]),
    ("web-05", "玄鸟安全新开的红队实习岗位在哪个城市?",
     [{"title": "玄鸟安全招聘模型红队实习生", "url": "https://example.com/xn-redteam", "published": "2026-11-01",
       "snippet": "工作地点:成都,每周到岗 4 天,实习 6 个月。"}], [r"成都"]),
    ("web-06", "曜石数据最近一轮融资是什么轮次?",
     [{"title": "曜石数据完成 B+ 轮融资", "url": "https://example.com/ys-funding", "published": "2026-10-18",
       "snippet": "曜石数据宣布完成 B+ 轮融资,将用于数据智能平台研发。"}], [r"B\+\s*轮"]),
    ("web-07", "北辰电商风控笔试据说考几道 SQL 题?",
     [{"title": "北辰电商风控岗笔试经验", "url": "https://example.com/bc-exam", "published": "2026-10-27",
       "snippet": "笔试 90 分钟,3 道 SQL 题和 2 道概率题。"}], [r"(?<!\d)3\s*道"]),
    ("web-08", "青檀医疗发布的医疗大模型叫什么?",
     [{"title": "青檀医疗发布医疗大模型 QT-Med", "url": "https://example.com/qt-med", "published": "2026-10-22",
       "snippet": "青檀医疗发布面向病历理解的医疗大模型 QT-Med 2。"}], [r"QT-?Med"]),
]

WEB_GOLD = {'web-01': '要求 1 年以上经验', 'web-02': 'LZ-Bench', 'web-03': '820 道任务', 'web-04': '2026 年 11 月 25 日', 'web-05': '成都', 'web-06': 'B+ 轮', 'web-07': '3 道 SQL 题', 'web-08': 'QT-Med 2'}

def main():
    with open(os.path.join(HERE, "fixtures", "jobs_fixture.json"), "w", encoding="utf-8") as f:
        json.dump({"synthetic": True, "note": "虚构公司,仅用于评测", "rows": JOBS}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "fixtures", "search_replay.jsonl"), "w", encoding="utf-8") as f:
        for tid, _, results, _ in WEB:
            f.write(json.dumps({"task_id": tid, "synthetic": True, "results": results}, ensure_ascii=False) + "\n")
    tasks = []
    for tid, src, q, pats, gold in notes_tasks:
        tasks.append({"id": tid, "tool": "notes_search", "question": q, "answer_all": pats,
                      "gold_text": gold, "source_rag_id": src})
    all_cos = [j["company"] for j in JOBS]
    for tid, q, pats in jobs_tasks():
        t = {"id": tid, "tool": "jobs_query", "question": q, "answer_all": pats}
        if pats[0] in all_cos:   # 答案是公司集合:其余公司不许出现,防止"全列出来"也判对
            t["answer_none"] = [c for c in all_cos if c not in pats]
            t["gold_text"] = "、".join(pats)
        else:
            t["gold_text"] = f"共 {COUNTS[tid]} 个"
        tasks.append(t)
    for tid, q, _, pats in WEB:
        tasks.append({"id": tid, "tool": "web_search", "question": q, "answer_all": pats, "gold_text": WEB_GOLD[tid]})
    with open(os.path.join(HERE, "tasks_v0.jsonl"), "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"{len(JOBS)} 条岗位,{len(tasks)} 道题")

if __name__ == "__main__":
    main()
