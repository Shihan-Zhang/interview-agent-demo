"""BFCL 风格判分的几种典型情况。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "fc"))
from fc_grader import grade  # noqa: E402
from summarize_fc import mcnemar_exact  # noqa: E402

SIMPLE = {"category": "simple_python",
          "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]}
PARALLEL = {"category": "parallel", "ground_truth": [
    {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
    {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}}]}


def test_optional_param_may_be_omitted_and_numbers_compare_by_value():
    assert grade(SIMPLE, [("calculate_triangle_area", {"base": 10.0, "height": 5})])[0]


def test_wrong_value_and_extra_param_fail():
    assert not grade(SIMPLE, [("calculate_triangle_area", {"base": 10, "height": 6})])[0]
    assert not grade(SIMPLE, [("calculate_triangle_area", {"base": 10, "height": 5, "color": "red"})])[0]


def test_parallel_is_order_insensitive_and_strings_are_normalised():
    calls = [("spotify.play", {"artist": "maroon 5", "duration": 15}),
             ("spotify.play", {"artist": "Taylor Swift", "duration": 20})]
    assert grade(PARALLEL, calls)[0]


def test_nested_dict_ground_truth_is_expanded():
    # 来自 simple_python_260:v1 把正确答案误判为错(4 个配置全错)
    item = {"category": "simple_python", "ground_truth": [{"paint_requirement.calculate": {
        "area": [{"width": [20], "height": [12]}], "paint_coverage": [350],
        "exclusion": [{"type": ["window"], "area": [15]}]}}]}
    good = {"area": {"height": 12, "width": 20}, "paint_coverage": 350, "exclusion": {"type": "window", "area": 15}}
    assert grade(item, [("paint_requirement.calculate", good)])[0]
    bad = {**good, "area": {"height": 12, "width": 21}}
    assert not grade(item, [("paint_requirement.calculate", bad)])[0]


def test_explicit_empty_string_matches_optional_marker():
    # 来自 live_multiple_74-34-0:可接受列表只有 "",模型显式传了 ""
    item = {"category": "live_multiple", "ground_truth": [{"f": {"topic": ["tech"], "subtopic": [""]}}]}
    assert grade(item, [("f", {"topic": "tech", "subtopic": ""})])[0]
    assert grade(item, [("f", {"topic": "tech"})])[0]
    assert not grade(item, [("f", {"topic": "tech", "subtopic": "ai"})])[0]


def test_irrelevance_requires_no_call():
    item = {"category": "irrelevance", "ground_truth": []}
    assert grade(item, [])[0]
    assert not grade(item, [("determine_body_mass_index", {"weight": 1, "height": 2})])[0]


def test_mcnemar_exact_matches_hand_calculation():
    assert abs(mcnemar_exact(6, 1) - 0.125) < 1e-12


def _canonical(allowed):
    a = next(x for x in allowed if x != "")
    if isinstance(a, dict):
        return {k: _canonical(v) for k, v in a.items() if any(x != "" for x in v)}
    if isinstance(a, list) and any(isinstance(x, dict) for x in a):
        return [_canonical([x]) for x in a]
    return a


def test_every_ground_truth_passes_its_own_grader():
    """自洽检查:把每道题的标准答案(取第一个可接受值,反序)当作模型输出,必须判对。
    v1 判分器在 simple_python_260 上的嵌套字典 bug,这个测试能直接抓到。"""
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "eval", "datasets", "bfcl_subset_v2.jsonl")
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    failures = []
    for it in items:
        if "relevance" in it["category"]:
            continue
        calls = [(name, {k: _canonical(v) for k, v in ps.items() if any(x != "" for x in v)})
                 for g in it["ground_truth"] for name, ps in g.items()][::-1]
        ok, reason = grade(it, calls)
        if not ok:
            failures.append((it["id"], reason))
    assert not failures, failures
