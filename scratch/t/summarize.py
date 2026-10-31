import json, collections, sys
d = json.load(open(sys.argv[1]))
g = collections.defaultdict(list)
for r in d:
    v = r['impl']['scipy 1.18.1']
    sev = v.get('severity')
    if sev is None:
        continue
    if sev == 'ok':
        continue
    g[(r['method'], tuple(r['shapes']), sev)].append((r['arg'], r['ref'], v['value'], v['ulp'], r.get('cond')))
for k in sorted(g, key=lambda k: (k[0], k[1], k[2])):
    pts = g[k]
    args = sorted(p[0] for p in pts)
    worst = max(pts, key=lambda p: (p[3] if p[3] == p[3] else 1e300))
    print(k, len(pts), 'args', args[:3], '...', args[-3:], 'worst', worst)
