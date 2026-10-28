import json, math, sys
from collections import defaultdict
rs=json.load(open('scratch/mielke/run3.json'))
key=lambda r: (r['method'], tuple(r['shapes']))
g=defaultdict(lambda: defaultdict(list))
for r in rs:
    im=r['impl']['scipy 1.18.1']
    g[key(r)][im['severity']].append(r)
for k in sorted(g, key=lambda k:(k[0],k[1])):
    d=g[k]
    cnt={s:len(v) for s,v in d.items() if s!='ok'}
    if not cnt: 
        print(k,'clean, n_ok',len(d.get('ok',[])) ); continue
    print('===',k[0],k[1],cnt)
    for s in ['loss','severe','moderate','minor']:
        v=d.get(s,[])
        if not v: continue
        args=sorted(x['arg'] for x in v)
        print('   ',s,'n=%d'%len(v),'args %.4g .. %.4g'%(args[0],args[-1]))
        worst=max(v,key=lambda x: (x['impl']['scipy 1.18.1']['ulp'] if x['impl']['scipy 1.18.1']['ulp']==x['impl']['scipy 1.18.1']['ulp'] else -1))
        im=worst['impl']['scipy 1.18.1']
        print('      worst arg=%r val=%r ref=%r ulp=%.3g cond=%.3g conds=%.3g'%(worst['arg'],im['value'],worst['ref'],im['ulp'],worst['cond'],worst['cond_shapes']))
