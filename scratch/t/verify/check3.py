# hyp2f1 identity for the lower tail: I_z(a,1/2) = z^a (1-z)^(1/2) / (a B(a,1/2)) 2F1(a+1/2, 1; a+1; z)
import sys; sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/scratch/t/verify')
import mpmath as mp
import vref as V
def tail_h(x, df):
    x = mp.mpf(x); df = mp.mpf(df); a = df/2; b = mp.mpf(1)/2
    z = df/(df + x*x)
    return z**a*mp.sqrt(1-z)/(a*mp.beta(a, b))*mp.hyp2f1(a+b, 1, a+1, z)/2
df274 = 2.7433514990818093
for q, df in [(1e-308, df274), (5.6234132519034905e-226, df274), (3.162277660168379e-151, df274),
              (0.29, 0.001), (1e-20, 0.1), (0.42999999999999994, 0.001), (0.205, 2.9)]:
    res = []
    for dps in (100, 200):
        mp.mp.dps = dps
        x = V.ppf(q, df)
        res.append((x, tail_h(-abs(x), df)))
    (x1, h1), (x2, h2) = res
    print(q, df, mp.nstr(-abs(x2), 20), 'hyp rel', mp.nstr(abs(h2-mp.mpf(q))/mp.mpf(q), 3),
          'h100/h200', mp.nstr(abs(h1-h2)/h2, 3))
