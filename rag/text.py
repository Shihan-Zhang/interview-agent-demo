"""
中文文本工具:句子切分、token 数估算、BM25 分词。

token 数是估算值(不加载 embedding 模型的分词器,保持切分过程轻量):
中日韩字符按 1 个 token 计,英文单词和数字按 1.3 个 token 计。
bge-m3 用的 XLM-R 分词器对中文大约 1–1.5 个字一个 token,所以这个估算偏保守(估多不估少)。
"""

import os
import re

_CJK = re.compile(r"[㐀-鿿豈-﫿]")
_WORD = re.compile(r"[A-Za-z0-9_]+")
# 句末标点(中英文)或换行之后切开;标点留在前一句
_SENT_END = re.compile(r"(?<=[。！？；!?;])|\n")

USERDICT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "userdict.txt")
STOPWORDS = set("的 了 是 在 和 与 及 或 也 都 就 而 被 把 对 为 以 于 上 下 中 这 那 个 一 不 有 我 你 他 她 它 们 吗 呢 吧 啊 "
                "什么 怎么 如何 哪些 哪个 为什么 可以 应该 the a an of to in is are and or for on with what how why".split())


def estimate_tokens(text):
    return len(_CJK.findall(text)) + int(1.3 * len(_WORD.findall(text)) + 0.5)


def split_sentences(text, offset=0):
    """把一段文字切成句子,返回 [(start, end)](绝对位置,含句末标点),跳过空白句子。"""
    spans, start = [], 0

    def add(s, e):
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.append((offset + s, offset + e))

    for m in _SENT_END.finditer(text):
        add(start, m.start())
        start = m.end() if text[m.start():m.end()] == "\n" else m.start()
    add(start, len(text))
    return spans


_jieba = None


def _load_jieba():
    global _jieba
    if _jieba is None:
        import jieba
        jieba.setLogLevel(60)
        if os.path.exists(USERDICT):
            with open(USERDICT, encoding="utf-8") as f:
                for line in f:
                    term = line.strip()
                    if term and not term.startswith("#"):
                        jieba.add_word(term, freq=10_000)   # 词频给大,保证专有名词整体切出
        _jieba = jieba
    return _jieba


def tokenize(text):
    """BM25 用的分词:jieba 搜索引擎模式 + 自定义词典,转小写,去停用词和纯标点。"""
    jieba = _load_jieba()
    out = []
    for w in jieba.lcut_for_search(text):
        w = w.strip().lower()
        if w and w not in STOPWORDS and re.search(r"\w", w):
            out.append(w)
    return out
