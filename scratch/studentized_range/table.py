import json, sys
for f in sys.argv[1:]:
    r = json.load(open(f))
    for rec in r:
        i = rec['impl']['scipy 1.18.1']
        if i['severity'] in ('ok',) and rec['region'] != 'bulk' and abs(float(rec['arg'])) > 1e-200 and float(rec['arg']) < 1e200 and False:
            continue
        a = float(rec['arg'])
        if not (1e-30 <= a <= 1e30): continue
        print(f"{rec['method']:4} {str(tuple(rec['shapes'])):18} {a:<22.6g} ref={float(rec['ref']) if rec['ref'] else 0:<24.16g} sp={float(i['value']):<24.16g} ulp={float(i['ulp']) if i['ulp'] else float('nan'):<10.3g} {i['severity']:8} cond={rec['cond']}")
