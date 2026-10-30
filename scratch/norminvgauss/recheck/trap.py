import sys, time
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/refs')
sys.path.insert(0, r'C:/Users/chbau/Documents/research/dist-accuracy/.claude/skills/dist-accuracy-review/scripts')
import mpmath as mp
import norminvgauss_ref as R
mp.mp.dps = 60
a, b = mp.mpf(10), mp.mpf('9.99999')
b = mp.mpf(9.99999)
x = mp.mpf(67.445903186346489)
g = mp.sqrt((a+b)*(a-b))
f = lambda s: R._L(s, x, a, b, g, 1)
sp, w, Lp = R._peak(f, -mp.log(g))
print('peak s', mp.nstr(sp, 10), 'w', mp.nstr(w, 10), 'transition s=log(x/b)', mp.nstr(mp.log(x/b), 10))
for hf in (20, 40, 80, 160, 320):
    t = time.time()
    print(hf, mp.nstr(R._mix_tail(x, a, b, 1, hfac=hf), 25), round(time.time()-t, 1))
# independent: mp.quad in s with breakpoints around transition and peak
with mp.workdps(40):
    F = lambda s: mp.exp(f(s))
    st = mp.log(x/b)
    pts = [-60, sp-10, st-2, st-0.5, st-0.1, st, st+0.1, st+0.5, st+2, sp, sp+5, sp+15, 20]
    pts = sorted(set(pts))
    print('quad', mp.nstr(mp.quad(F, pts), 25))
