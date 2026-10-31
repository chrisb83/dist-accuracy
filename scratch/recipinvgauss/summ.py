import json, sys
from collections import defaultdict
d = json.load(open(sys.argv[1]))
g = defaultdict(list)
for p in d:
    s = p['impl']['scipy 1.18.1']
    if s['severity'] in ('loss', 'severe', 'moderate', 'minor'):
        g[(p['method'], p['shapes'][0])].append((p['arg'], p['ref'], s['value'], s['ulp'], s['severity'], p['cond']))
for k, v in sorted(g.items()):
    from collections import Counter
    print(k, dict(Counter(x[4] for x in v)))
    for x in v:
        print('   %r ref=%r got=%r ulp=%.3g %s cond=%.3g' % x)
