import json,collections,sys
d=json.load(open(sys.argv[1]))
g=collections.defaultdict(list)
for r in d:
    v=r['impl']['scipy 1.18.1']
    if v['severity'] in ('ok',): continue
    g[(r['method'],tuple(r['shapes']))].append((r['arg'],r['ref'],v['value'],v['ulp'],r['cond'],v['severity']))
for k,L in sorted(g.items()):
    c=collections.Counter(x[5] for x in L)
    print(k, dict(c))
    L.sort(key=lambda t:t[0])
    for t in (L if len(L)<=6 else L[:3]+L[-3:]): print('   arg=%r ref=%.6g val=%r ulp=%.3g cond=%.3g %s'%t)
