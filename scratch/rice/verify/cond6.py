# Conditioning and implied q error for the F6 points (normalized quad of the pdf)
import mpmath as mp
mp.mp.dps = 40
pdf = lambda x, b: x*mp.exp(-(x*x+b*b)/2)*mp.besseli(0, x*b)
def sf(x, b):
    c = pdf(x, b); return c*mp.quad(lambda t: pdf(t, b)/c, [x, x+mp.mpf(1)/4, x+1, x+4, x+60])
for q, b, xr, xs in [(0.9999999999999997, 15, '23.103234311158455', 22.930016816055357),
                     (0.9999999999999997, 40, '48.08795632026136', 47.811524755353474),
                     (0.9999999999522942, 40, '46.485633208514183', 46.48562540459734)]:
    q, b, xr, xs = mp.mpf(q), mp.mpf(b), mp.mpf(xr), mp.mpf(xs)
    print(mp.nstr(q/(xr*pdf(xr, b)), 3), '1-q =', mp.nstr(1-q, 5), 'sf(x_scipy) =', mp.nstr(sf(xs, b), 5), 'diff/eps =', mp.nstr((sf(xs, b)-(1-q))/mp.mpf(2)**-52, 3))
