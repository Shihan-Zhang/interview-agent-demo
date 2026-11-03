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


def test_irrelevance_requires_no_call():
    item = {"category": "irrelevance", "ground_truth": []}
    assert grade(item, [])[0]
    assert not grade(item, [("determine_body_mass_index", {"weight": 1, "height": 2})])[0]


def test_mcnemar_exact_matches_hand_calculation():
    assert abs(mcnemar_exact(6, 1) - 0.125) < 1e-12
