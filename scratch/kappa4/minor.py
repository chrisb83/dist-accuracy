import json, collections
recs=json.load(open('scratch/kappa4/points.json'))
out=collections.defaultdict(list)
for r in recs:
    v=list(r['impl'].values())[0]
    if v['severity']=='minor':
        out[(r['method'],tuple(r['shapes']))].append((v['ulp'],r['arg'],v['value'],r['ref'],r['cond'],r['region']))
for k in sorted(out):
    rows=sorted(out[k],reverse=True)
    print('==',k,len(rows))
    for u,a,val,ref,c,reg in rows[:2]:
        print('   ulp=%-8.3g arg=%-22r scipy=%-22r ref=%-22r cond=%-8.3g %s'%(u,a,val,ref,c,reg))
