import json, sys, collections
d=json.load(open(sys.argv[1]))
order={'loss':4,'severe':3,'moderate':2,'minor':1,'ok':0}
g=collections.defaultdict(list)
for p in d:
    s=p['impl']['scipy 1.18.1']
    g[(p['method'],tuple(p['shapes']))].append((order[s['severity']],s['ulp'] if s['ulp'] is not None else float('inf'),p['arg'],p['ref'],s['value'],p['cond'],s['severity']))
for k in sorted(g):
    L=g[k]; c=collections.Counter(x[6] for x in L)
    w=max(L,key=lambda x:(x[0],x[1]))
    if w[0]==0: continue
    print(k[0],k[1],dict(c),'| worst arg=%r ref=%r val=%r ulp=%.3g cond=%.3g'%(w[2],w[3],w[4],w[1],w[5]))
