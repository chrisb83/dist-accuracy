import json, sys, collections
d = json.load(open(sys.argv[1]))
g = collections.defaultdict(list)
for p in d:
    i = p["impl"]["scipy 1.18.1"]
    g[repr(p["shapes"][0])].append((p["arg"], p["ref"], i["value"], i["ulp"], i["severity"], p["cond"]))
for k, v in g.items():
    sev = collections.Counter(r[4] for r in v)
    print(k, dict(sev))
    for r in v:
        if r[4] != "ok":
            print("   x=%-24r ref=%-24r sp=%-24r ulp=%.3g %s cond=%.2g" % r)
