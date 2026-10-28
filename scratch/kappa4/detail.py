import json, collections, math, sys
recs=json.load(open('scratch/kappa4/points.json'))
meth=sys.argv[1] if len(sys.argv)>1 else None
out=collections.defaultdict(list)
for r in recs:
    v=list(r['impl'].values())[0]
    if v['severity'] in ('loss','severe','moderate'):
        out[(r['method'],tuple(r['shapes']))].append((r['arg'],v['value'],r['ref'],v['ulp'],r['cond'],v['severity'],r['region']))
for k in sorted(out):
    if meth and k[0]!=meth: continue
    rows=out[k]
    rows.sort(key=lambda t: abs(t[0]))
    print('==',k, len(rows))
    sel = rows[:3]+rows[len(rows)//2:len(rows)//2+2]+rows[-3:]
    seen=set()
    for a,val,ref,u,c,s,reg in sel:
        if a in seen: continue
        seen.add(a)
        print('   arg=%-24r scipy=%-24r ref=%-24r ulp=%-9.3g cond=%-9.3g %s %s'%(a,val,ref,u,c,s,reg))
