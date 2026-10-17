"""
最小 Agent demo 的工具层。
每个工具 = 一个真正会执行、返回真实结果的函数,不是 LLM 编出来的。
"""

import math

# ---------- 工具 1: 计算器 ----------
# 对应 Agent 架构里的"精确计算"场景——LLM 自己做算术容易出错,
# 该交给确定性代码做的事,就不要让模型"脑补"。

def calculator(expression: str) -> dict:
    """
    计算一个数学表达式。只允许数字和 + - * / ** ( ) sqrt,防止任意代码执行。
    """
    allowed_names = {"sqrt": math.sqrt, "abs": abs, "round": round}
    try:
        # 白名单字符检查,避免 eval 被用来执行任意代码
        allowed_chars = set("0123456789.+-*/() ")
        stripped = expression.replace("sqrt", "").replace(" ", "")
        if not set(stripped) <= (allowed_chars - {" "}):
            return {"ok": False, "error": f"表达式包含不允许的字符: {expression}"}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------- 工具 2: 统计/机器学习概念查询 ----------
# 对应 RAG 的最简化版本:真实 RAG 是"向量检索",这里先用关键词精确匹配
# 模拟"检索外部知识库"这个动作——循环逻辑和真实 RAG 完全一样,
# 只是把"向量相似度检索"换成了"关键词查字典",方便你先把 Agent 主循环跑通。
# 之后升级成真 RAG,只需要替换这个函数的内部实现,不用动 agent.py 的循环逻辑。

_KNOWLEDGE_BASE = {
    "bradley-terry": "Bradley-Terry 模型:通过两两对战的胜负记录,估计每个对象的潜在实力值。"
                      "P(A胜B) = p_A / (p_A + p_B)。可改写为逻辑回归求解。"
                      "典型应用:Chatbot Arena 的模型排名。",
    "p值": "p值:在原假设成立的前提下,观测到当前结果或更极端结果的概率。"
           "p值小,说明当前数据在原假设下很罕见,倾向于拒绝原假设。p值不是'原假设为真的概率'。",
    "cuped": "CUPED (Controlled-experiment Using Pre-Experiment Data):"
             "利用实验开始前的历史数据作为协变量,通过回归调整降低实验指标的方差,"
             "在不增加样本量的情况下缩短所需实验周期。由微软实验平台团队提出。",
    "srm": "SRM (Sample Ratio Mismatch,样本比例失配):AB实验中实际分流比例"
           "与预设比例不符,通常用卡方拟合优度检验检测。出现SRM说明实验本身有偏差,"
           "结果不可信,需要先排查埋点/随机化问题,而不是直接看显著性结果。",
    "位置偏差": "位置偏差 (Position Bias):LLM-as-Judge 场景下,裁判模型会系统性偏好"
               "排在特定位置(通常是第一或第二)的回答,与回答本身质量无关。"
               "常见修正方法:把每对回答按两种顺序各跑一次,只有两次结果一致才算真实胜负。",
    "语义熵": "语义熵 (Semantic Entropy):幻觉检测方法之一。对同一问题多次采样生成回答,"
             "先按语义(而非字面)聚类,再对语义簇计算熵。熵越高,说明模型在语义空间中"
             "游移不定,越可能是幻觉。",
}


def concept_lookup(term: str) -> dict:
    """
    在统计/机器学习概念知识库中查找一个术语的定义。
    模拟 RAG 的检索步骤(这里用关键词匹配代替向量检索)。
    """
    key = term.strip().lower()
    for k, v in _KNOWLEDGE_BASE.items():
        if k in key or key in k:
            return {"ok": True, "term": k, "definition": v}
    return {
        "ok": False,
        "error": f"知识库中没有找到与'{term}'匹配的词条。"
                 f"当前收录: {', '.join(_KNOWLEDGE_BASE.keys())}",
    }


# ---------- 工具 schema(Anthropic 格式),供 Claude 的 tool use API 使用 ----------

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "计算一个数学表达式的精确结果。适用于任何需要算术运算的场景,"
                        "不要自己心算,交给这个工具保证准确。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式,例如 '0.95 ** 8' 或 'sqrt(16) + 3'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "concept_lookup",
        "description": "查询统计学/机器学习/AB测试/模型评测相关术语的定义。"
                        "当用户问到一个专有名词,而你不确定准确定义或想确认细节时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "要查询的术语,例如 'CUPED'、'p值'、'Bradley-Terry'",
                }
            },
            "required": ["term"],
        },
    },
]

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "concept_lookup": concept_lookup,
}


def _to_openai_tool_schema(anthropic_schema: dict) -> dict:
    """
    把 Anthropic 风格的工具描述({name, description, input_schema})
    转换成 OpenAI 风格({type: function, function: {name, description, parameters}})。
    两家的"参数长什么样"这部分(JSON Schema)是通用的,只是外面包的壳子名字不一样——
    这就是为什么只需要写一个转换函数,而不用给每个工具重复定义两套schema。
    """
    return {
        "type": "function",
        "function": {
            "name": anthropic_schema["name"],
            "description": anthropic_schema["description"],
            "parameters": anthropic_schema["input_schema"],
        },
    }


# OpenAI兼容接口(NVIDIA NIM、以及任何其他OpenAI协议的供应商)用这份
OPENAI_TOOL_SCHEMAS = [_to_openai_tool_schema(s) for s in TOOL_SCHEMAS]
