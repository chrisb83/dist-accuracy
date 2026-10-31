import json, sys
recs = json.load(open(sys.argv[1]))
for r in recs:
    v = r['impl']['scipy 1.18.1']
    if v['severity'] != 'ok':
        print(f"{r['method']:7s} {r['shapes'][0]:<10.4g} {r['arg']:<24.17g} ref={r['ref']!r:<26} sp={v['value']!r:<26} ulp={v['ulp']:.3g} cond={r['cond']:.3g} cs={r['cond_shapes']:.3g} {v['severity']}")
