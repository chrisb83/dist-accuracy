import json, sys, collections
d = json.load(open(sys.argv[1]))
want = sys.argv[2:]  # optional methods
groups = collections.defaultdict(list)
for p in d:
    i = p['impl']['scipy 1.18.1']
    if i['severity'] in ('ok',) or i['severity'] is None: continue
    if want and p['method'] not in want: continue
    groups[(p['method'], tuple(p['shapes']), i['severity'], p['region'])].append(p)
for k, v in sorted(groups.items()):
    v.sort(key=lambda p: p['arg'])
    w = max(v, key=lambda p: p['impl']['scipy 1.18.1']['ulp'] if p['impl']['scipy 1.18.1']['ulp']==p['impl']['scipy 1.18.1']['ulp'] else 1e300)
    print(k, len(v), 'arg %.4g..%.4g' % (v[0]['arg'], v[-1]['arg']), 'worst', w['arg'], w['ref'], w['impl']['scipy 1.18.1']['value'], w['impl']['scipy 1.18.1']['ulp'], 'cond', w['cond'])
