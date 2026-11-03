import json, sys
from collections import defaultdict
d=json.load(open(sys.argv[1]))
g=defaultdict(list)
for p in d:
    i=p['impl']['scipy 1.18.1']
    if i['severity'] in ('ok',): continue
    g[(p['method'],tuple(p['shapes']))].append((p['arg'],p['ref'],i['value'],i['ulp'],i['severity'],p['cond']))
for k,v in g.items():
    sev=defaultdict(int)
    for r in v: sev[r[4]]+=1
    args=sorted(r[0] for r in v)
    print(k, dict(sev), 'arg range %.3g..%.3g'%(args[0],args[-1]))
    for r in sorted(v, key=lambda r:r[0])[:: max(1,len(v)//6)]:
        print('    arg=%.17g ref=%.17g val=%r ulp=%.3g %s cond=%.3g'%r)
