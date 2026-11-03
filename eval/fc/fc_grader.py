"""
BFCL 风格的函数调用判分(简化版 AST 匹配,规则参照 BFCL 官方实现,细节可能有差异)。

只判"模型发出的函数调用对不对",不执行函数:
- simple / multiple:恰好一次调用;函数名正确;参数逐个比对;
- parallel:调用次数和标准答案相同,顺序无关,逐一配对;
- irrelevance:给的函数都和问题无关,正确做法是不调用任何函数。

参数比对:
- 标准答案里每个参数给出"可接受取值列表",列表里有 "" 表示这个参数可以不传;
- 字符串:去掉空格和常见标点、转小写后比较(和 BFCL 一样,"New York City" 与 "new york city" 视为相同);
- 数字:数值相等即可(5 和 5.0 视为相同);
- 列表、字典:逐元素递归比较;
- 模型传了标准答案里没有的参数,判错。
"""

import itertools
import re

_STRIP = re.compile(r"[ ,./\-_*^'\"]")


def _norm(v):
    if isinstance(v, str):
        return _STRIP.sub("", v).lower()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _value_ok(value, allowed):
    nv = _norm(value)
    return any(a != "" and _norm(a) == nv for a in allowed)


def match_call(name, args, expected):
    """expected 形如 {"函数名": {"参数": [可接受取值...]}};返回 (是否匹配, 原因)。"""
    (exp_name, exp_params), = expected.items()
    if name != exp_name:
        return False, f"函数名错误:{name} ≠ {exp_name}"
    extra = set(args) - set(exp_params)
    if extra:
        return False, f"多传了参数:{sorted(extra)}"
    for p, allowed in exp_params.items():
        if p not in args:
            if "" not in allowed:
                return False, f"缺少必需参数:{p}"
            continue
        if not _value_ok(args[p], allowed):
            return False, f"参数 {p} 的值 {args[p]!r} 不在可接受范围 {allowed}"
    return True, "ok"


def grade(item, calls):
    """calls:[(函数名, 参数 dict)],函数名已换回原名。"""
    cat, gt = item["category"], item["ground_truth"]
    if cat == "irrelevance":
        return (not calls), ("正确地没有调用函数" if not calls else f"不应调用函数,却调用了 {calls[0][0]}")
    if len(calls) != len(gt):
        return False, f"调用次数 {len(calls)} ≠ 期望 {len(gt)}"
    if len(gt) == 1:
        return match_call(calls[0][0], calls[0][1], gt[0])
    for perm in itertools.permutations(range(len(calls))):          # parallel:顺序无关
        if all(match_call(calls[i][0], calls[i][1], gt[j])[0] for j, i in enumerate(perm)):
            return True, "ok"
    return False, "并行调用与标准答案无法一一对应:" + "; ".join(
        match_call(c[0], c[1], g)[1] for c, g in zip(calls, gt))
