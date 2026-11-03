import json, sys, math
from collections import defaultdict
d = json.load(open(sys.argv[1]))
groups = defaultdict(list)
for r in d:
    v = r['impl']['scipy 1.18.1']
    if v['severity'] != 'ok':
        groups[(r['method'], tuple(r['shapes']))].append((r['arg'], r['ref'], v['value'], v['ulp'], r['cond'], v['severity']))
for k in sorted(groups):
    g = groups[k]
    sev = defaultdict(int)
    for p in g: sev[p[5]] += 1
    print(k, dict(sev))
    for p in sorted(g):
        print('   arg=%-24r ref=%-24r got=%-24r ulp=%-9.3g cond=%-9.3g %s' % p)
