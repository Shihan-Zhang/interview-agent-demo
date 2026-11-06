"""
汇总 toolmaze_run.py 的调用日志:token、思考占比、缓存命中、延迟、接口错误、截断,并按单价估算花费。

    python eval/agent_error/cost_report.py <calls.jsonl> --in-price 0.24 --out-price 1.80 [--episodes 20 --scale 2000]
单价为美元 / 百万 token(硅基流动 Qwen3.5-35B-A3B 官方博客价;国内控制台以人民币计,以账户显示为准)。
"""
import argparse
import json
import statistics as st


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--in-price", type=float, default=0.24)
    ap.add_argument("--out-price", type=float, default=1.80)
    ap.add_argument("--episodes", type=int, default=None, help="日志覆盖的回合数,用于外推")
    ap.add_argument("--scale", type=int, default=None, help="外推到多少回合")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.log, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("ok")]
    err = [r for r in rows if not r.get("ok")]
    pin = sum(r.get("prompt_tokens") or 0 for r in ok)
    pout = sum(r.get("completion_tokens") or 0 for r in ok)
    rea = sum(r.get("reasoning_tokens") or 0 for r in ok)
    cached = sum(r.get("cached_tokens") or 0 for r in ok)
    lat = [r["latency_s"] for r in ok]
    cost = pin / 1e6 * a.in_price + pout / 1e6 * a.out_price
    print(f"请求 {len(rows)}(成功 {len(ok)},失败 {len(err)})")
    print(f"输入 {pin:,} token(缓存命中 {cached:,});输出 {pout:,} token,其中思考 {rea:,}"
          f"({rea / pout:.0%})" if pout else "")
    print(f"有 reasoning_content 的回复 {sum(r.get('has_reasoning_content') for r in ok)} / {len(ok)};"
          f"finish_reason=length(输出被截断){sum(r.get('finish_reason') == 'length' for r in ok)}")
    if lat:
        print(f"延迟 中位数 {st.median(lat):.1f}s,p95 {pct(lat, .95):.1f}s,最长 {max(lat):.1f}s")
    print(f"估算花费 ${cost:.3f}(不计缓存折扣)")
    if err:
        kinds = {}
        for r in err:
            k = r["error"].split(":")[0]
            kinds[k] = kinds.get(k, 0) + 1
        print("错误类型:", kinds)
    if a.episodes and a.scale:
        print(f"外推到 {a.scale} 回合:约 ${cost / a.episodes * a.scale:.1f}")


if __name__ == "__main__":
    main()
