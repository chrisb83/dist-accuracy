import mpmath as mp
import importlib.util, sys
x, c = 1.0000000000000002, 1e-17
def z(x,c): x=mp.mpf(x); c=mp.mpf(c); return (x-1)/(c*mp.sqrt(x))
def pdf(t,c): c=mp.mpf(c); return (t+1)/(2*c*mp.sqrt(2*mp.pi*t**3))*mp.exp(-(t-1)**2/(2*t*c**2))
out = {}
for d in (100, 200):
    mp.mp.dps = d
    X = mp.mpf(x); C = mp.mpf(c)
    e = mp.erfc(z(x,c)/mp.sqrt(2))/2
    q = mp.quad(lambda t: pdf(t, C), [X, X+C, X+3*C, X+10*C, X+30*C, X+100*C, mp.inf])
    out[d] = (e, q)
mp.mp.dps = 50
print('erfc 100 vs 200:', mp.nstr(abs(out[100][0]-out[200][0])/out[200][0], 3))
print('quad 100 vs 200:', mp.nstr(abs(out[100][1]-out[200][1])/out[200][1], 3))
print('quad200 vs erfc200:', mp.nstr(abs(out[200][1]-out[200][0])/out[200][0], 3), mp.nstr(out[200][0], 20))
