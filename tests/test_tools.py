"""
单元测试:只测"这个函数本身对不对",不涉及Agent循环、不涉及任务完成得好不好。
每个测试函数只验证一件事,函数名就是"验证的是什么"——这样测试挂了,
光看函数名就知道是哪个行为坏了,不用去读测试内容猜。
"""

from tools import calculator, concept_lookup


def test_calculator_basic_arithmetic():
    result = calculator("2 + 2")
    assert result["ok"] is True
    assert result["result"] == 4


def test_calculator_with_sqrt():
    result = calculator("sqrt(16) + 3")
    assert result["ok"] is True
    assert result["result"] == 7.0


def test_calculator_rejects_disallowed_characters():
    # 防止 eval() 被拿来执行任意代码——这是个安全测试,不是功能测试
    result = calculator("__import__('os').system('echo hacked')")
    assert result["ok"] is False


def test_calculator_handles_division_by_zero_gracefully():
    # 关键:除零应该被 try/except 捕获,返回 ok=False,而不是让程序崩溃退出
    result = calculator("1/0")
    assert result["ok"] is False
    assert "error" in result


def test_concept_lookup_finds_known_term():
    result = concept_lookup("CUPED")
    assert result["ok"] is True
    assert result["term"] == "cuped"


def test_concept_lookup_is_case_insensitive():
    result = concept_lookup("cuped")
    assert result["ok"] is True


def test_concept_lookup_reports_missing_term_as_failure():
    # 关键:查不到不应该返回一个假的/编造的定义,应该明确说"没找到"
    result = concept_lookup("一个完全不存在的术语XYZ123")
    assert result["ok"] is False
    assert "error" in result
