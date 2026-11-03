import json, sys
from collections import defaultdict
d = json.load(open('scratch/argus/points.json'))
g = defaultdict(list)
for p in d:
    i = p['impl']['scipy 1.18.1']
    if i['severity'] != 'ok':
        g[(p['method'], p['shapes'][0], i['severity'])].append((p['arg'], p['ref'], i['value'], i['ulp'], p['cond']))
for k in sorted(g):
    v = g[k]
    args = sorted(a[0] for a in v)
    w = max(v, key=lambda t: (t[3] if t[3]==t[3] else 1e300))
    print(k, len(v), 'args %.3g..%.17g' % (args[0], args[-1]), 'worst', w)
