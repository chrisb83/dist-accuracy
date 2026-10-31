import json, sys, collections
recs = json.load(open(sys.argv[1]))
skip_pdf_big = len(sys.argv) > 2
g = collections.defaultdict(list)
for r in recs:
    v = r["impl"]["scipy 1.18.1"]
    if v["severity"] in ("ok",): continue
    sh = tuple(r["shapes"])
    if skip_pdf_big and r["method"] == "pdf" and sh[0] in (1e6, 1e12, 300.0): continue
    g[(r["method"], sh)].append((r["arg"], r["ref"], v["value"], v["ulp"], v["severity"], r["cond"]))
for (m, sh), L in sorted(g.items()):
    c = collections.Counter(x[4] for x in L)
    xs = sorted(x[0] for x in L)
    worst = max(L, key=lambda t: (t[3] if t[3] == t[3] else 1e300))
    print(f"{m} {sh}: {dict(c)}  x in [{xs[0]:.3g}, {xs[-1]:.3g}]  worst x={worst[0]:.4g} ref={worst[1]:.6g} got={worst[2]:.6g} ulp={worst[3]:.2g} cond={worst[5]:.3g}")
    for t in sorted(L):
        print(f"     x={t[0]:<11.4g} ref={t[1]:<24.17g} got={t[2]:<24.17g} ulp={t[3]:<9.2g} {t[4]:<8} cond={t[5]:.3g}")
