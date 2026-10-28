import json, collections, math
recs=json.load(open('scratch/kappa4/points.json'))
key=lambda r:(r['method'],tuple(r['shapes']))
g=collections.defaultdict(lambda: collections.Counter())
worst={}
for r in recs:
    v=list(r['impl'].values())[0]
    s=v['severity']
    g[key(r)][s]+=1
    if s in ('loss','severe','moderate','minor'):
        k=(key(r),s)
        if k not in worst or (v['ulp']>worst[k][1]['ulp'] if math.isfinite(v['ulp']) else False):
            worst.setdefault(k,(r,v))
for k in sorted(g):
    c=g[k]
    bad={s:c[s] for s in ('loss','severe','moderate','minor') if c[s]}
    if bad:
        print(k, dict(c))
