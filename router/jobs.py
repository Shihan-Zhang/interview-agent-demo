"""
岗位表:结构化查询,不走 RAG。

为什么不让模型直接写 SQL(text-to-SQL):
1. 安全:只读也可能被拼出全表扫描或注入;参数化的过滤条件只能做我们允许的事;
2. 小模型更稳:4B 模型填几个枚举参数,比写对一条带中文列名的 SQL 容易得多;
3. 报错可操作:参数不合法时能告诉模型"可选值是哪些",模型据此改参数重试
   (Anthropic 工具设计建议:错误信息要能指导下一步,而不是只给一个错误码)。

数据来源:Notion「求职追踪 Job Tracker」导出的快照(个人数据,不进仓库,放 data/jobs/)。
"""

import json
from datetime import date

CITIES = ["上海", "北京", "杭州", "深圳", "其他"]
STATUSES = ["待投递", "已投递", "笔试/测评", "一面", "二面", "三面/HR面", "等offer", "Offer", "已拒/挂", "放弃"]
TRACKS = ["电商风控", "互联网数据分析", "金融风控", "医疗AI/健康数据", "药企/RWE", "其他"]
RECRUIT_TYPES = ["社招", "校招", "应届社招/补录", "实习"]
LEVELS = ["高", "中", "低"]
SORT_KEYS = ["applied_date", "next_date", "priority"]
_LEVEL_RANK = {"高": 0, "中": 1, "低": 2}

# Notion 列名 -> 内部字段名
NOTION_FIELDS = {
    "公司": "company", "岗位": "title", "城市": "city", "状态": "status", "赛道": "tracks",
    "招聘类型": "recruit_type", "优先级": "priority", "匹配度": "match", "投递渠道": "channel",
    "applied": "applied_date", "next_date": "next_date", "薪资": "salary", "JD要点": "jd",
}


def normalize_notion_row(row: dict) -> dict:
    out = {NOTION_FIELDS[k]: v for k, v in row.items() if k in NOTION_FIELDS}
    tracks = out.get("tracks")
    if isinstance(tracks, str):                      # Notion SQL 把多选导出成 JSON 字符串
        out["tracks"] = json.loads(tracks) if tracks.strip() else []
    out.setdefault("tracks", [])
    out["tracks"] = out["tracks"] or []
    return out


def load_jobs(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["rows"] if isinstance(data, dict) else data
    return [normalize_notion_row(r) if "公司" in r else r for r in rows]


def _check_enum(name, value, allowed):
    if value is not None and value not in allowed:
        return {"ok": False, "error_type": "invalid_arguments",
                "error": f"参数 {name}='{value}' 不合法,可选值: {allowed}"}
    return None


def _check_date(name, value):
    if value is None:
        return None, None
    try:
        return date.fromisoformat(value), None
    except (TypeError, ValueError):
        return None, {"ok": False, "error_type": "invalid_arguments",
                      "error": f"参数 {name}='{value}' 不是 YYYY-MM-DD 格式的日期"}


class JobsTable:
    def __init__(self, rows: list):
        self.rows = rows

    def query(self, status=None, city=None, track=None, recruit_type=None, priority=None,
              keyword=None, applied_from=None, applied_to=None, next_before=None,
              sort_by=None, limit=10):
        for name, value, allowed in [("status", status, STATUSES), ("city", city, CITIES),
                                     ("track", track, TRACKS), ("recruit_type", recruit_type, RECRUIT_TYPES),
                                     ("priority", priority, LEVELS), ("sort_by", sort_by, SORT_KEYS)]:
            err = _check_enum(name, value, allowed)
            if err:
                return err
        bounds = {}
        for name, value in [("applied_from", applied_from), ("applied_to", applied_to),
                            ("next_before", next_before)]:
            bounds[name], err = _check_date(name, value)
            if err:
                return err
        try:
            limit = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            return {"ok": False, "error_type": "invalid_arguments", "error": f"limit='{limit}' 不是整数"}

        def keep(r):
            if status and r.get("status") != status:
                return False
            if city and r.get("city") != city:
                return False
            if track and track not in r.get("tracks", []):
                return False
            if recruit_type and r.get("recruit_type") != recruit_type:
                return False
            if priority and r.get("priority") != priority:
                return False
            if keyword:
                hay = " ".join(str(r.get(k) or "") for k in ("company", "title", "jd")).lower()
                if keyword.lower() not in hay:
                    return False
            ad = r.get("applied_date")
            if bounds["applied_from"] and (not ad or date.fromisoformat(ad) < bounds["applied_from"]):
                return False
            if bounds["applied_to"] and (not ad or date.fromisoformat(ad) > bounds["applied_to"]):
                return False
            nd = r.get("next_date")
            if bounds["next_before"] and (not nd or date.fromisoformat(nd) > bounds["next_before"]):
                return False
            return True

        hits = [r for r in self.rows if keep(r)]
        if sort_by == "priority":
            hits.sort(key=lambda r: _LEVEL_RANK.get(r.get("priority"), 9))
        elif sort_by in ("applied_date", "next_date"):
            # 没有日期的行排最后,而不是让 None 和字符串比较报错
            hits.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by) or ""))
        fields = ("company", "title", "city", "status", "tracks", "recruit_type", "priority",
                  "applied_date", "next_date")
        return {"ok": True, "count_total": len(hits), "returned": min(limit, len(hits)),
                "rows": [{k: r.get(k) for k in fields} for r in hits[:limit]]}
