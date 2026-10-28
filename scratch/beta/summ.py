import json, sys, math
d = json.load(open(sys.argv[1]))
from collections import defaultdict
g = defaultdict(list)
for r in d:
    i = r['impl']['scipy 1.18.1']
    if i['severity'] in ('loss', 'severe', 'moderate'):
        g[(r['method'], tuple(r['shapes']), i['severity'])].append((r['arg'], r['ref'], i['value'], i['ulp'], r['cond']))
for k, v in sorted(g.items()):
    args = [a for a, *_ in v]
    w = max(v, key=lambda t: (t[3] if t[3] == t[3] else 1e400))
    print(k, len(v), 'arg range', min(args), max(args), 'worst', w)
