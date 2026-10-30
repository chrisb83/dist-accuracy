import json, collections
d = json.load(open(r"C:\Users\chbau\Documents\research\dist-accuracy\scratch\gausshyper\points.json"))
recs = d if isinstance(d, list) else d.get("points", d)
by = collections.defaultdict(list)
for r in recs:
    v = list(r["impl"].values())[0]
    by[str(r["shapes"])].append((r["arg"], r["ref"], v["value"], v["ulp"], v["severity"], r.get("cond")))
for k, L in by.items():
    print(k)
    cnt = collections.Counter(x[4] for x in L)
    print("  ", dict(cnt))
    for x in sorted(L, key=lambda t: -(t[3] if t[3] == t[3] else -1))[:8]:
        print("   ", x)
