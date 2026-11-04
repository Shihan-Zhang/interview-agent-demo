"""
切分:Markdown 文档 → 带标题路径的块 → rag/store/chunks.jsonl

步骤(每一步都保留在原文里的字符位置,评测时按原文位置判断命中,不依赖块编号):
1. 按标题切成小节,记录标题路径(代码块里以 # 开头的行不当标题);
2. 小节内切成"单元":代码块整体一个单元(公式不拆开),连续的表格行整体一个单元,其余按句子切;
3. 单元依次装进块,估算 token 数不超过 max_tokens;相邻块重叠最后 overlap 个句子单元;
4. 块开头拼上标题路径的最后两级(零成本的 Contextual Retrieval)。
   不用完整路径:一级标题是文档名,每块都带会让所有块在这个词上同样相似,稀释排序,也浪费 token。

运行:python -m rag.ingest data/notes            (目录下所有 .md)
"""

import argparse
import hashlib
import json
import os
import re
import sys

try:
    from rag.text import estimate_tokens, split_sentences
except ImportError:  # 直接运行脚本时
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag.text import estimate_tokens, split_sentences

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, "rag", "store")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def parse_sections(text):
    """返回 [(标题路径列表, 正文起点, 正文终点)]。路径不含一级标题(文档名),一级标题单独返回。"""
    sections, stack, title = [], [], None
    in_fence, body_start, pos = False, 0, 0
    cur_path = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        m = None if in_fence or stripped.startswith("```") else _HEADING.match(line.rstrip("\n"))
        if m:
            sections.append((list(cur_path), body_start, pos))
            level, name = len(m.group(1)), m.group(2).strip()
            if level == 1:
                title, stack = name, []
            else:
                stack = [(lv, n) for lv, n in stack if lv < level] + [(level, name)]
            cur_path = [n for _, n in stack]
            body_start = pos + len(line)
        pos += len(line)
    sections.append((list(cur_path), body_start, pos))
    return title, [(p, s, e) for p, s, e in sections if text[s:e].strip()]


def split_units(text, start, end):
    """小节正文 → 单元 [(kind, s, e)],kind 为 code / table / sentence。"""
    units = []
    lines = []
    p = start
    for line in text[start:end].splitlines(keepends=True):
        lines.append((p, p + len(line), line))
        p += len(line)
    buf_start = None  # 普通文字的起点
    k = 0
    while k < len(lines):
        s, e, line = lines[k]
        st = line.strip()
        if st.startswith("```") or st.startswith("|"):
            if buf_start is not None:
                units += [("sentence", a, b) for a, b in split_sentences(text[buf_start:s], buf_start)]
                buf_start = None
            if st.startswith("```"):                      # 代码块:到配对的 ``` 为止
                j = k + 1
                while j < len(lines) and not lines[j][2].strip().startswith("```"):
                    j += 1
                j = min(j, len(lines) - 1)
            else:                                         # 表格:连续的 | 开头行
                j = k
                while j + 1 < len(lines) and lines[j + 1][2].strip().startswith("|"):
                    j += 1
            u_end = lines[j][1]
            while u_end > s and text[u_end - 1].isspace():
                u_end -= 1
            units.append(("code" if st.startswith("```") else "table", s, u_end))
            k = j + 1
            continue
        if buf_start is None:
            buf_start = s
        k += 1
    if buf_start is not None:
        units += [("sentence", a, b) for a, b in split_sentences(text[buf_start:end], buf_start)]
    return units


def _split_long(text, kind, s, e, max_tokens):
    """单个单元超长时硬切:表格按行切并在每段前补表头,其余按字符切。返回 [(片段文字, s, e)]。"""
    seg = text[s:e]
    if estimate_tokens(seg) <= max_tokens:
        return [(seg, s, e)]
    if kind == "table":
        rows = seg.splitlines(keepends=True)
        header = "".join(rows[:2])
        out, cur, cur_s, p = [], "", s, s
        for row in rows:
            if cur and estimate_tokens(header + cur + row) > max_tokens:
                out.append(((header if cur_s != s else "") + cur, cur_s, p))
                cur, cur_s = "", p
            cur += row
            p += len(row)
        out.append(((header if cur_s != s else "") + cur, cur_s, e))
        return [(t.rstrip(), a, b) for t, a, b in out]
    out, a = [], s
    step = max(1, len(seg) * max_tokens // max(estimate_tokens(seg), 1))
    while a < e:
        b = min(e, a + step)
        out.append((text[a:b], a, b))
        a = b
    return out


def chunk_document(text, source, max_tokens=400, overlap=1, min_tokens=8):
    title, sections = parse_sections(text)
    chunks = []
    for path, s, e in sections:
        units = []
        for kind, a, b in split_units(text, s, e):
            units += [(kind, t, x, y) for t, x, y in _split_long(text, kind, a, b, max_tokens)]
        i = 0
        while i < len(units):
            j, toks = i, 0
            while j < len(units) and (j == i or toks + estimate_tokens(units[j][1]) <= max_tokens):
                toks += estimate_tokens(units[j][1])
                j += 1
            body = "\n".join(u[1] for u in units[i:j])
            prefix = " > ".join(path[-2:])
            cid = hashlib.sha1(f"{source}:{units[i][2]}:{units[j - 1][3]}".encode()).hexdigest()[:12]
            if toks < min_tokens and i == 0 and j >= len(units):
                break                      # 整个小节只有零星几个字(如文档开头的日期署名行),不成块
            chunks.append({"id": cid, "source": source, "doc_title": title, "path": path,
                           "start": units[i][2], "end": units[j - 1][3], "text": body,
                           "embed_text": f"[{prefix}] {body}" if prefix else body,
                           "tokens_est": toks})
            if j >= len(units):
                break
            # 重叠:下一块从最后 overlap 个句子单元开始(代码、表格不重复)
            back = 0
            while back < overlap and j - 1 - back > i and units[j - 1 - back][0] == "sentence":
                back += 1
            i = j - back
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="Markdown 文件所在目录")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(STORE, "chunks.jsonl"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.corpus) for f in fs if f.endswith(".md"))
    all_chunks = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        rel = os.path.relpath(path, args.corpus).replace("\\", "/")
        cs = chunk_document(text, rel, args.max_tokens, args.overlap)
        all_chunks += cs
        print(f"{rel}: {len(text)} 字符 → {len(cs)} 块")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    toks = sorted(c["tokens_est"] for c in all_chunks)
    if toks:
        print(f"共 {len(all_chunks)} 块;估算 token 中位数 {toks[len(toks) // 2]},最大 {toks[-1]} → {args.out}")


if __name__ == "__main__":
    main()
